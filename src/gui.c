#include "gui.h"
#include <SDL3/SDL.h>
#include <stdio.h>
#include <math.h>
#include "model.h"

// Save the 28x28 image with its digit label
static void save_image(float *image)
{
    int label;
    printf("\nEnter the digit label (0-9): ");
    scanf("%d",&label);

    if(label<0 || label>9){
        printf("Invalid label!\n");
        return;
    }

    char filename[100];
    int image_number=1;

    while(1){
        sprintf(filename,"drawings/%d_%03d.pgm",label,image_number);
        FILE *check=fopen(filename,"rb");

        if(check==NULL)
            break;

        fclose(check);
        image_number++;
    }

    FILE *file=fopen(filename,"wb");

    if(file==NULL){
        printf("Could not save image: %s\n",filename);
        return;
    }

    fprintf(file,"P5\n");
    fprintf(file,"28 28\n");
    fprintf(file,"255\n");

    for(int i=0;i<784;i++){
        unsigned char pixel=(unsigned char)(image[i]*255.0f);
        fwrite(&pixel,1,1,file);
    }

    fclose(file);
    printf("Image saved: %s\n",filename);
}

// Clear the drawing canvas
static void clear_canvas(float *canvas,float *image,int *prediction,float *probabilities)
{
    for(int i=0;i<CANVAS_SIZE*CANVAS_SIZE;i++)
        canvas[i]=0.0f;

    for(int i=0;i<IMAGE_SIZE;i++)
        image[i]=0.0f;

    *prediction=-1;

    for(int i=0;i<10;i++)
        probabilities[i]=0.0f;
}

// Find the bounding box of the drawing
static void find_bounding_box(float *canvas,int *min_x,int *min_y,int *max_x,int *max_y)
{
    *min_x=CANVAS_SIZE;
    *min_y=CANVAS_SIZE;
    *max_x=-1;
    *max_y=-1;

    for(int y=0;y<CANVAS_SIZE;y++){
        for(int x=0;x<CANVAS_SIZE;x++){
            if(canvas[y*CANVAS_SIZE+x]>0.0f){
                if(x<*min_x) *min_x=x;
                if(x>*max_x) *max_x=x;
                if(y<*min_y) *min_y=y;
                if(y>*max_y) *max_y=y;
            }
        }
    }
}

// Convert the canvas into a centered 28x28 MNIST image
static void canvas_to_image(float *canvas,float *image)
{
    int min_x,min_y,max_x,max_y;

    find_bounding_box(canvas,&min_x,&min_y,&max_x,&max_y);

    for(int i=0;i<IMAGE_SIZE;i++)
        image[i]=0.0f;

    if(max_x==-1)
        return;

    int margin=10;

    min_x-=margin;
    min_y-=margin;
    max_x+=margin;
    max_y+=margin;

    if(min_x<0) min_x=0;
    if(min_y<0) min_y=0;
    if(max_x>=CANVAS_SIZE) max_x=CANVAS_SIZE-1;
    if(max_y>=CANVAS_SIZE) max_y=CANVAS_SIZE-1;

    int width=max_x-min_x+1;
    int height=max_y-min_y+1;

    int target_size=20;
    float scale;

    if(width>height)
        scale=(float)target_size/width;
    else
        scale=(float)target_size/height;

    int new_width=(int)(width*scale+0.5f);
    int new_height=(int)(height*scale+0.5f);

    if(new_width<1) new_width=1;
    if(new_height<1) new_height=1;

    int start_x=(28-new_width)/2;
    int start_y=(28-new_height)/2;

    for(int y=0;y<new_height;y++){
        for(int x=0;x<new_width;x++){
            float sx1=min_x+(float)x/scale;
            float sy1=min_y+(float)y/scale;
            float sx2=min_x+(float)(x+1)/scale;
            float sy2=min_y+(float)(y+1)/scale;

            int ix1=(int)sx1;
            int iy1=(int)sy1;
            int ix2=(int)sx2;
            int iy2=(int)sy2;

            if(ix2<=ix1) ix2=ix1+1;
            if(iy2<=iy1) iy2=iy1+1;

            float sum=0.0f;
            int count=0;

            for(int sy=iy1;sy<iy2;sy++){
                for(int sx=ix1;sx<ix2;sx++){
                    if(sx>=0 && sx<CANVAS_SIZE && sy>=0 && sy<CANVAS_SIZE){
                        sum+=canvas[sy*CANVAS_SIZE+sx];
                        count++;
                    }
                }
            }

            if(count>0){
                int px=start_x+x;
                int py=start_y+y;

                if(px>=0 && px<28 && py>=0 && py<28)
                    image[py*28+px]=sum/count;
            }
        }
    }

    float total_mass=0.0f;
    float center_x=0.0f;
    float center_y=0.0f;

    for(int y=0;y<28;y++){
        for(int x=0;x<28;x++){
            float value=image[y*28+x];
            total_mass+=value;
            center_x+=x*value;
            center_y+=y*value;
        }
    }

    if(total_mass<=0.0f)
        return;

    center_x/=total_mass;
    center_y/=total_mass;

    float target_x=13.5f;
    float target_y=13.5f;

    int shift_x=(int)(target_x-center_x);
    int shift_y=(int)(target_y-center_y);

    float centered[IMAGE_SIZE]={0};

    for(int y=0;y<28;y++){
        for(int x=0;x<28;x++){
            int nx=x+shift_x;
            int ny=y+shift_y;

            if(nx>=0 && nx<28 && ny>=0 && ny<28)
                centered[ny*28+nx]=image[y*28+x];
        }
    }

    for(int i=0;i<IMAGE_SIZE;i++)
        image[i]=centered[i];
}

// Check if image contains valid numbers
static int image_is_valid(float *image)
{
    for(int i=0;i<IMAGE_SIZE;i++){
        if(!isfinite(image[i])){
            printf("INPUT INVALID AT %d:%f\n",i,image[i]);
            return 0;
        }
    }

    return 1;
}

// Run the neural network
static int predict_digit(float *image,Network *network,float *probabilities)
{
    if(!image_is_valid(image))
        return -1;

    network_forward(network,image);

    int output_layer=network->layer_count;
    int prediction=0;
    float highest=network->activations[output_layer][0];

    for(int i=0;i<10;i++){
        probabilities[i]=network->activations[output_layer][i];

        if(!isfinite(probabilities[i])){
            printf("OUTPUT INVALID AT %d:%f\n",i,probabilities[i]);
            return -1;
        }
    }

    for(int i=1;i<10;i++){
        if(probabilities[i]>highest){
            highest=probabilities[i];
            prediction=i;
        }
    }

    printf("\nPredicted:%d\n",prediction);
    printf("Confidence:%.2f%%\n",highest*100.0f);
    printf("PROBABILITIES:\n");

    for(int i=0;i<10;i++)
        printf("%d:%.2f%%\n",i,probabilities[i]*100.0f);

    return prediction;
}

// Train using the current drawing and selected label
static void train_from_drawing(float *image,Network *network,int label)
{
    if(label<0 || label>9)
        return;

    float targets[10]={0};
    targets[label]=1.0f;

    int epochs=10;
    float learning_rate=0.003f;

    printf("\n=============================\n");
    printf("TRAINING FROM DRAWING\n");
    printf("=============================\n");
    printf("Correct digit: %d\n",label);
    printf("Training...\n");

    for(int epoch=0;epoch<epochs;epoch++){
        network_forward(network,image);
        network_backward(network,targets,learning_rate);
        network_update(network,learning_rate,1);
        printf("Training epoch %d/%d complete\n",epoch+1,epochs);
    }

    printf("Training complete!\n");

    if(model_save(network,"models/interactive_model.bin"))
        printf("Model saved: models/interactive_model.bin\n");
    else
        printf("Could not save model!\n");

    printf("=============================\n\n");
}

// Draw the predicted digit using seven segments
static void draw_seven_segment(SDL_Renderer *renderer,int digit,int panel_x,int panel_y)
{
    int segments[10][7]={
        {1,1,1,1,1,1,0},
        {0,1,1,0,0,0,0},
        {1,1,0,1,1,0,1},
        {1,1,1,1,0,0,1},
        {0,1,1,0,0,1,1},
        {1,0,1,1,0,1,1},
        {1,0,1,1,1,1,1},
        {1,1,1,0,0,0,0},
        {1,1,1,1,1,1,1},
        {1,1,1,1,0,1,1}
    };

    if(digit<0 || digit>9)
        return;

    int x=panel_x+100;
    int y=panel_y+25;
    int w=50;
    int h=100;
    int t=8;

    SDL_SetRenderDrawColor(renderer,255,255,255,255);

    if(segments[digit][0]){
        SDL_FRect r={(float)x,(float)y,(float)w,(float)t};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][1]){
        SDL_FRect r={(float)(x+w-t),(float)y,(float)t,(float)(h/2)};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][2]){
        SDL_FRect r={(float)(x+w-t),(float)(y+h/2),(float)t,(float)(h/2)};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][3]){
        SDL_FRect r={(float)x,(float)(y+h-t),(float)w,(float)t};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][4]){
        SDL_FRect r={(float)x,(float)(y+h/2),(float)t,(float)(h/2)};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][5]){
        SDL_FRect r={(float)x,(float)y,(float)t,(float)(h/2)};
        SDL_RenderFillRect(renderer,&r);
    }

    if(segments[digit][6]){
        SDL_FRect r={(float)x,(float)(y+h/2-t/2),(float)w,(float)t};
        SDL_RenderFillRect(renderer,&r);
    }
}

// Draw the prediction panel
static void draw_prediction_panel(SDL_Renderer *renderer,int prediction,float *probabilities)
{
    int panel_x=620;
    int panel_y=80;

    if(prediction==-1)
        return;

    SDL_SetRenderDrawColor(renderer,255,255,255,255);

    SDL_FRect box={(float)panel_x,(float)panel_y,250.0f,150.0f};
    SDL_RenderRect(renderer,&box);

    draw_seven_segment(renderer,prediction,panel_x,panel_y);

    float confidence=probabilities[prediction];

    if(!isfinite(confidence)) confidence=0.0f;
    if(confidence<0.0f) confidence=0.0f;
    if(confidence>1.0f) confidence=1.0f;

    SDL_SetRenderDrawColor(renderer,70,70,70,255);

    SDL_FRect background={(float)panel_x,250.0f,250.0f,15.0f};
    SDL_RenderFillRect(renderer,&background);

    SDL_SetRenderDrawColor(renderer,0,255,0,255);

    SDL_FRect bar={(float)panel_x,250.0f,confidence*250.0f,15.0f};
    SDL_RenderFillRect(renderer,&bar);
}

// Draw probability bars
static void draw_probability_bars(SDL_Renderer *renderer,float *probabilities,int prediction)
{
    int bars_x=620;
    int bars_y=280;
    int bar_width=250;
    int bar_height=25;

    for(int i=0;i<10;i++){
        int y=bars_y+i*35;

        SDL_SetRenderDrawColor(renderer,70,70,70,255);
        SDL_FRect background={(float)bars_x,(float)y,(float)bar_width,(float)bar_height};
        SDL_RenderFillRect(renderer,&background);

        float probability=probabilities[i];

        if(!isfinite(probability)) probability=0.0f;
        if(probability<0.0f) probability=0.0f;
        if(probability>1.0f) probability=1.0f;

        int width=(int)(probability*bar_width);

        if(i==prediction)
            SDL_SetRenderDrawColor(renderer,0,255,0,255);
        else
            SDL_SetRenderDrawColor(renderer,180,180,180,255);

        SDL_FRect bar={(float)bars_x,(float)y,(float)width,(float)bar_height};
        SDL_RenderFillRect(renderer,&bar);

        SDL_SetRenderDrawColor(renderer,255,255,255,255);
        char text[20];
        sprintf(text,"%d  %.1f%%",i,probability*100.0f);
        SDL_RenderDebugText(renderer,(float)(bars_x+5),(float)(y+7),text);
    }
}

// Draw the 0-9 training buttons
static void draw_training_buttons(SDL_Renderer *renderer,int training_mode)
{
    int start_x=620;
    int start_y=630;
    int button_width=45;
    int button_height=30;
    int gap=5;

    SDL_SetRenderDrawColor(renderer,255,255,255,255);

    SDL_RenderDebugText(renderer,620.0f,610.0f,training_mode ? "TRAINING - CHOOSE CORRECT DIGIT" : "T = TRAIN");

    for(int i=0;i<10;i++){
        int row=i/5;
        int col=i%5;

        int x=start_x+col*(button_width+gap);
        int y=start_y+row*(button_height+gap);

        SDL_FRect button={(float)x,(float)y,(float)button_width,(float)button_height};

        if(training_mode)
            SDL_SetRenderDrawColor(renderer,100,100,100,255);
        else
            SDL_SetRenderDrawColor(renderer,50,50,50,255);

        SDL_RenderFillRect(renderer,&button);

        SDL_SetRenderDrawColor(renderer,255,255,255,255);
        SDL_RenderRect(renderer,&button);

        char text[2];
        sprintf(text,"%d",i);

        SDL_RenderDebugText(renderer,(float)(x+19),(float)(y+9),text);
    }
}

// Draw the main canvas
static void draw_canvas(SDL_Renderer *renderer,float *canvas,float mouse_x,float mouse_y,int drawing)
{
    SDL_SetRenderDrawColor(renderer,25,25,25,255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer,255,255,255,255);

    for(int y=0;y<CANVAS_SIZE;y++){
        for(int x=0;x<CANVAS_SIZE;x++){
            if(canvas[y*CANVAS_SIZE+x]>0.0f){
                SDL_FRect pixel={(float)x,(float)y,1.0f,1.0f};
                SDL_RenderFillRect(renderer,&pixel);
            }
        }
    }

    if(drawing){
        SDL_FRect brush={mouse_x-6,mouse_y-6,12,12};
        SDL_RenderFillRect(renderer,&brush);
    }
}

// Print the 28x28 image
static void print_image(float *image)
{
    printf("\n28x28 IMAGE\n");

    for(int y=0;y<28;y++){
        for(int x=0;x<28;x++){
            float value=image[y*28+x];

            if(value>0.5f)
                printf("#");
            else if(value>0.1f)
                printf("+");
            else
                printf(" ");
        }

        printf("\n");
    }
}

// Check if mouse is inside a training button
static int get_training_button(int x,int y)
{
    int start_x=620;
    int start_y=630;
    int button_width=45;
    int button_height=30;
    int gap=5;

    for(int i=0;i<10;i++){
        int row=i/5;
        int col=i%5;

        int bx=start_x+col*(button_width+gap);
        int by=start_y+row*(button_height+gap);

        if(x>=bx && x<=bx+button_width && y>=by && y<=by+button_height)
            return i;
    }

    return -1;
}

int gui_run(float *image,Network *network)
{
    printf("1 Entered gui_run()\n");

    if(!SDL_Init(SDL_INIT_VIDEO)){
        printf("2 SDL initialization failed: %s\n",SDL_GetError());
        return 0;
    }

    printf("2 SDL initialized successfully!\n");

    SDL_Window *window=NULL;
    SDL_Renderer *renderer=NULL;

    printf("3 Creating window...\n");

    if(!SDL_CreateWindowAndRenderer("Digit Recognizer",1000,700,0,&window,&renderer)){
        printf("4 Window creation failed: %s\n",SDL_GetError());
        SDL_Quit();
        return 0;
    }

    printf("4 Window created successfully!\n");

    int running=1;
    int drawing=0;
    int training_mode=0;

    float mouse_x=0.0f;
    float mouse_y=0.0f;

    float canvas[CANVAS_SIZE*CANVAS_SIZE]={0};
    float probabilities[10]={0};

    int prediction=-1;

    while(running){
        SDL_Event event;

        while(SDL_PollEvent(&event)){
            if(event.type==SDL_EVENT_QUIT)
                running=0;

            if(event.type==SDL_EVENT_KEY_DOWN){
                printf("Key pressed:%d\n",event.key.key);

                if(event.key.key==SDLK_ESCAPE)
                    running=0;

                // Clear everything
                if(event.key.key==SDLK_C){
                    clear_canvas(canvas,image,&prediction,probabilities);
                    training_mode=0;
                    printf("Canvas Cleared!\n");
                }

                // Reset prediction
                if(event.key.key==SDLK_R){
                    prediction=-1;

                    for(int i=0;i<10;i++)
                        probabilities[i]=0.0f;

                    printf("Prediction reset!\n");
                }

                // Save drawing
                if(event.key.key==SDLK_S){
                    if(image_is_valid(image))
                        save_image(image);
                    else
                        printf("Nothing to save!\n");
                }

                // Enter runs recognition
                if(event.key.key==SDLK_RETURN){
                    int min_x,min_y,max_x,max_y;

                    find_bounding_box(canvas,&min_x,&min_y,&max_x,&max_y);

                    if(max_x==-1){
                        printf("Canvas Empty!\n");
                    }
                    else{
                        printf("Bounding box: (%d,%d) -> (%d,%d)\n",min_x,min_y,max_x,max_y);

                        canvas_to_image(canvas,image);
                        print_image(image);

                        printf("\nRunning neural network...\n");

                        prediction=predict_digit(image,network,probabilities);
                    }
                }

                // T enters training mode
                if(event.key.key==SDLK_T){
                    int min_x,min_y,max_x,max_y;

                    find_bounding_box(canvas,&min_x,&min_y,&max_x,&max_y);

                    if(max_x==-1){
                        printf("Canvas Empty! Draw a digit first.\n");
                    }
                    else{
                        canvas_to_image(canvas,image);
                        training_mode=1;
                        printf("Training mode enabled. Choose a digit in the GUI.\n");
                    }
                }
            }

            if(event.type==SDL_EVENT_MOUSE_MOTION){
                mouse_x=event.motion.x;
                mouse_y=event.motion.y;

                if(drawing && !training_mode){
                    int x=(int)mouse_x;
                    int y=(int)mouse_y;
                    int radius=6;

                    for(int dy=-radius;dy<=radius;dy++){
                        for(int dx=-radius;dx<=radius;dx++){
                            int px=x+dx;
                            int py=y+dy;

                            if(px>=0 && px<CANVAS_SIZE && py>=0 && py<CANVAS_SIZE){
                                if(dx*dx+dy*dy<=radius*radius)
                                    canvas[py*CANVAS_SIZE+px]=1.0f;
                            }
                        }
                    }
                }
            }

            if(event.type==SDL_EVENT_MOUSE_BUTTON_DOWN){
                if(event.button.button==SDL_BUTTON_LEFT){
                    int x=(int)event.button.x;
                    int y=(int)event.button.y;

                    if(training_mode){
                        int label=get_training_button(x,y);

                        if(label!=-1){
                            printf("Selected training label: %d\n",label);

                            train_from_drawing(image,network,label);

                            prediction=predict_digit(image,network,probabilities);

                            training_mode=0;
                        }
                    }
                    else{
                        if(x>=0 && x<CANVAS_SIZE && y>=0 && y<CANVAS_SIZE)
                            drawing=1;
                    }
                }
            }

            if(event.type==SDL_EVENT_MOUSE_BUTTON_UP){
                if(event.button.button==SDL_BUTTON_LEFT)
                    drawing=0;
            }
        }

        draw_canvas(renderer,canvas,mouse_x,mouse_y,drawing);
        draw_prediction_panel(renderer,prediction,probabilities);
        draw_probability_bars(renderer,probabilities,prediction);
        draw_training_buttons(renderer,training_mode);

        SDL_RenderPresent(renderer);
    }

    printf("5 Closing window...\n");

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 1;
}
