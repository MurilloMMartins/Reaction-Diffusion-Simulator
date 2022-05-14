#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "./includes/shader.h"
#include "./includes/stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>

#include "includes/imgui_impl_glfw.h"
#include "includes/imgui_impl_opengl3.h"

void processInput(GLFWwindow *window);
float laplaceA(int y, int x, std::vector<std::pair<float, float>>& array);
float laplaceB(int y, int x, std::vector<std::pair<float, float>>& array);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

void splat(std::vector<std::pair<float, float>> &array, int x, int y, int size);
void resetConcentration();

//screen settings
const unsigned int SCR_WIDTH = 768;
const unsigned int SCR_HEIGHT = 768;

//reaction diffusion variables
int gridSizeX = 256;
int gridSizeY = 256;
float Da = 1.0f;
float Db = 0.5f;
float f = 0.012f;
float k = 0.053f;
float dt = 1.0f;
int splatSize = 10;
std::vector<std::pair<float, float>> concentration(gridSizeX * gridSizeY);


int main(){
    //initializing GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    //creating GLFW window
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Reaction Diffusion - Grey Scott Model", NULL, NULL);
    if(window == NULL){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    //initializing GLAD
    if(!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    //compiling shader program
    Shader reactionShader("./shaders/vertex.vs", "./shaders/fragment.fs");

    //plane vertex data (this plane will cover the entire screen)
    float vertices[] = {
        //position              //texture coords
        //first triangle
        -1.0f,  1.0f, 0.0f,     0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f,
         1.0f,  1.0f, 0.0f,     1.0f, 1.0f,
        //second triangle
         1.0f,  1.0f, 0.0f,     1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f,
         1.0f, -1.0f, 0.0f,     1.0f, 0.0f
    };

    //creating VAO, VBO and loading vertex data
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(0);

    //texture attibute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    resetConcentration();

    //creating simulation texture
    //this texture will be applied on our plane and show the simulation
    unsigned int colorChannels = 4;
    GLubyte *simulationTexture = new GLubyte[gridSizeY * gridSizeX * colorChannels];

    for(unsigned int i = 0; i < gridSizeY; i++){
        for(unsigned int j = 0; j < gridSizeX; j++){
            float f = concentration[i * gridSizeY + j].first;
            if(f < 0.0f) f = 0.0f;
            GLubyte color = f*255;
            simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 0] = color;   //red
            simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 1] = color;   //green
            simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 2] = color;   //blue
            simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 3] = 255;     //alpha
        }
    }

    //allocating texture in OpenGL
    GLuint texture1;
    glGenTextures(1, &texture1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);

    //setting texture's wrapping/filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gridSizeX, gridSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, simulationTexture);
    glGenerateMipmap(GL_TEXTURE_2D);

    //IMGUI initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();

    //render loop
    while(!glfwWindowShouldClose(window)){
        //processing input
        processInput(window);

        //setting background and clearing screen
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //using shader
        reactionShader.use();
        //deciding new concentration base on previous concentrations
        for(int i = 1; i < gridSizeY-1; i++){
            for(int j = 1; j < gridSizeX-1; j++){
                float a = concentration[i * gridSizeY + j].first;
                float b = concentration[i * gridSizeY + j].second;
                //updating values based on the Grey-Scott Model equations
                float A = a + ((Da * laplaceA(i, j, concentration)) - (a * b * b) + (f*(1 - a))*dt);
                float B = b + ((Db * laplaceB(i, j, concentration)) + (a * b * b) - (b*(k + f))*dt);

                //constraining the values
                if(A > 1.0f) A = 1.0f;
                if(A < 0.0f) A = 0.0f;
                if(B > 1.0f) B = 1.0f;
                if(B < 0.0f) B = 0.0f;

                concentration[i * gridSizeY + j].first = A;
                concentration[i * gridSizeY + j].second = B;
            }
        }

        //updating texture's values
        for(unsigned int i = 0; i < gridSizeY; i++){
            for(unsigned int j = 0; j < gridSizeX; j++){
                float f = concentration[i * gridSizeY + j].first - concentration[i * gridSizeY + j].second;
                if(f < 0.0f) f = 0.0f;
                if(f > 1.0f) f = 1.0f;
                GLubyte red = f*255;
                GLubyte green = f*255;
                GLubyte blue = f*255;
                simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 0] = red; //red
                simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 1] = green; //green
                simulationTexture[i * gridSizeY * colorChannels + j * colorChannels + 2] = blue;  //blue
            }
        }

        //updating texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gridSizeX, gridSizeY, GL_RGBA, GL_UNSIGNED_BYTE, simulationTexture);

        //starting ImGUI frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //rendering plane
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        //rendering GUI
        ImGui::Begin("Simulation Controls");
        if(ImGui::Button("Reset"))
            resetConcentration();
        ImGui::SliderFloat("Feed Rate", &f, 0.0f, 0.3f);
        ImGui::SliderFloat("Kill Rate", &k, 0.0f, 0.08f);
        ImGui::SliderInt("Splat size", &splatSize, 1, 25);
        ImGui::End();

        //render ImGUI to screen
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //updating buffers and polling events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    //deleting allocated array
    delete [] simulationTexture;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    

    //destroying GLFW objects
    glfwDestroyWindow(window);
    glfwTerminate();
    return(0);
}

/**
 * @brief function to process user input
 * 
 * @param window glfw window
 */
void processInput(GLFWwindow *window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

/**
 * @brief function to apply a Laplacian Kernel to the A chemical
 * 
 * @param y array Y's position
 * @param x array X's position
 * @param array pair vector
 * @return float new concentration
 */
float laplaceA(int y, int x, std::vector<std::pair<float, float>>& array){
    float total = 0;
    total += array[y * gridSizeY + x].first*(-1.0f);
    total += array[y * gridSizeY + (x-1)].first*0.2f;
    total += array[y * gridSizeY + (x+1)].first*0.2f;
    total += array[(y-1) * gridSizeY + (x)].first*0.2f;
    total += array[(y+1) * gridSizeY + (x)].first*0.2f;
    total += array[(y-1) * gridSizeY + (x-1)].first*0.05f;
    total += array[(y+1) * gridSizeY + (x-1)].first*0.05f;
    total += array[(y-1) * gridSizeY + (x+1)].first*0.05f;
    total += array[(y+1) * gridSizeY + (x+1)].first*0.05f;

    return total;
}

/**
 * @brief function to apply a Laplacian Kernel to the B chemical
 * 
 * @param y array Y's position
 * @param x array X's position
 * @param array pair vector
 * @return float new concentration
 */
float laplaceB(int y, int x, std::vector<std::pair<float, float>>& array){
    float total = 0;
    total += array[y * gridSizeY + x].second*(-1.0f);
    total += array[y * gridSizeY + (x-1)].second*0.2f;
    total += array[y * gridSizeY + (x+1)].second*0.2f;
    total += array[(y-1) * gridSizeY + (x)].second*0.2f;
    total += array[(y+1) * gridSizeY + (x)].second*0.2f;
    total += array[(y-1) * gridSizeY + (x-1)].second*0.05f;
    total += array[(y+1) * gridSizeY + (x-1)].second*0.05f;
    total += array[(y-1) * gridSizeY + (x+1)].second*0.05f;
    total += array[(y+1) * gridSizeY + (x+1)].second*0.05f;

    return total;
}

void splat(std::vector<std::pair<float, float>> &array, int x, int y, int size){
    for(int i = 0; i < gridSizeY; i++){
        for(int j = 0; j < gridSizeX; j++){
            if((i > y - size && j > x - size) && (i < y + size && j < x + size)){
                array[i * gridSizeY + j].first = 0.0f;
                array[i * gridSizeY + j].second = 1.0f;
            }
        }
    }
}

//function to change the size of the viewport in case the user changes the screen size
void framebuffer_size_callback(GLFWwindow *window, int width, int height){
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
        double xpos, ypos;
        double xRatio = SCR_WIDTH/gridSizeX;
        double yRatio = SCR_HEIGHT/gridSizeY;
        glfwGetCursorPos(window, &xpos, &ypos);
        splat(concentration, static_cast<int>(xpos)/xRatio, gridSizeY - static_cast<int>(ypos)/yRatio, splatSize);
    }   
}

void resetConcentration(){
    for(int i = 0; i < gridSizeY; i++){
        for(int j = 0; j < gridSizeX; j++){
            concentration[i * gridSizeY + j].first = 1.0f;
            concentration[i * gridSizeY + j].second = 0.0f;
        }
    }
}