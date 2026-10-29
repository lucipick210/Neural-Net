#include "gui.h"
#include <SDL3/SDL.h>
#include <stdio.h>
#include <math.h>

// Save the 28x28 image with its digit label
static void save_image(float *image,int number)
{
    int label;

    printf("\nEnter the digit label (0-9): ");
    scanf("%d",&label);

    if(label < 0 || label > 9){
        printf("Invalid label!\n");
        return;
    }

    char filename[100];

    sprintf(filename,"drawings/%d_%03d.pgm",label,number);

    FILE *file = fopen(filename,"wb");

    if(file == NULL){
        printf("Could not save image: %s\n",filename);
        return;
    }

    fprintf(file,"P5\n");
    fprintf(file,"28 28\n");
    fprintf(file,"255\n");

    for(int i =0;i<784;i++){
        unsigned char pixel = (unsigned char)(image[i] * 255.0f);

        fwrite(&pixel,1,1,file);
    }

    fclose(file);

    printf("Image saved: %s\n",filename);
}

// Clear the drawing canvas, image and prediction
static void clear_canvas(
    float *canvas,
    float *image,
    int *prediction,
    float *probabilities)
{
    for(int i =0;i<CANVAS_SIZE*CANVAS_SIZE;i++)
        canvas[i]=0.0f;

    for(int i =0;i<IMAGE_SIZE;i++)
        image[i]=0.0f;

    *prediction=-1;

    for(int i =0;i<10;i++)
        probabilities[i]=0.0f;
}


// Find the bounding box of the drawn digit
static void find_bounding_box(
    float *canvas,
    int *min_x,
    int *min_y,
    int *max_x,
    int *max_y)
{
    *min_x=CANVAS_SIZE;
    *min_y=CANVAS_SIZE;
    *max_x=-1;
    *max_y=-1;

    for(int y=0;y<CANVAS_SIZE;y++){
        for(int x=0;x<CANVAS_SIZE;x++){

            if(canvas[y*CANVAS_SIZE+x]>0.0f){

                if(x<*min_x)
                    *min_x=x;

                if(x>*max_x)
                    *max_x=x;

                if(y<*min_y)
                    *min_y=y;

                if(y>*max_y)
                    *max_y=y;
            }
        }
    }
}


// Convert the 560x560 canvas into a 28x28 MNIST image
static void canvas_to_image(float *canvas,float *image){

    int min_x;
    int min_y;
    int max_x;
    int max_y;

    find_bounding_box(
        canvas,
        &min_x,
        &min_y,
        &max_x,
        &max_y
    );

    for(int i=0;i<IMAGE_SIZE;i++)
        image[i]=0.0f;

    // Stop if the canvas is empty
    if(max_x==-1)
        return;


    // Add some space around the digit
    int margin=10;

    min_x-=margin;
    min_y-=margin;
    max_x+=margin;
    max_y+=margin;

    if(min_x<0)
        min_x=0;

    if(min_y<0)
        min_y=0;

    if(max_x>=CANVAS_SIZE)
        max_x=CANVAS_SIZE-1;

    if(max_y>=CANVAS_SIZE)
        max_y=CANVAS_SIZE-1;


    int width=max_x-min_x+1;
    int height=max_y-min_y+1;

    // Resize the digit to about 20x20 pixels
    int target_size=20;

    float scale;

    if(width>height)
        scale=(float)target_size/width;
    else
        scale=(float)target_size/height;

    int new_width=(int)(width*scale+0.5f);
    int new_height=(int)(height*scale+0.5f);

    if(new_width<1)
        new_width=1;

    if(new_height<1)
        new_height=1;


    // Put the digit in the center of the 28x28 image
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

            if(ix2<=ix1)
                ix2=ix1+1;

            if(iy2<=iy1)
                iy2=iy1+1;

            float sum=0.0f;
            int count=0;


            // Calculate the average brightness
            for(int sy=iy1;sy<iy2;sy++){
                for(int sx=ix1;sx<ix2;sx++){

                    if(
                        sx>=0 &&
                        sx<CANVAS_SIZE &&
                        sy>=0 &&
                        sy<CANVAS_SIZE)
                    {
                        sum+=canvas[sy*CANVAS_SIZE+sx];
                        count++;
                    }
                }
            }


            if(count>0){

                int px=start_x+x;
                int py=start_y+y;

                if(
                    px>=0 &&
                    px<28 &&
                    py>=0 &&
                    py<28)
                {
                    image[py*28+px]=sum/count;
                }
            }
        }
    }


    // Calculate the center of mass of the digit
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


    // MNIST digits are approximately centered
    float target_x=13.5f;
    float target_y=13.5f;

    int shift_x=(int)(target_x-center_x);
    int shift_y=(int)(target_y-center_y);

    float centered[IMAGE_SIZE]={0};


    // Move the digit closer to the center
    for(int y=0;y<28;y++){
        for(int x=0;x<28;x++){

            int nx=x+shift_x;
            int ny=y+shift_y;

            if(
                nx>=0 &&
                nx<28 &&
                ny>=0 &&
                ny<28)
            {
                centered[ny*28+nx]=image[y*28+x];
            }
        }
    }


    for(int i=0;i<IMAGE_SIZE;i++)
        image[i]=centered[i];
}


// Check if the input contains valid numbers
static int image_is_valid(float *image){

    for(int i=0;i<IMAGE_SIZE;i++){

        if(!isfinite(image[i])){

            printf(
                "INPUT INVALID AT %d:%f\n",
                i,
                image[i]
            );

            return 0;
        }
    }

    return 1;
}


// Run the neural network and get the predicted digit
static int predict_digit(
    float *image,
    Network *network,
    float *probabilities)
{
    if(!image_is_valid(image))
        return -1;


    // Run forward propagation
    network_forward(network,image);

    int output_layer=network->layer_count;

    int prediction=0;

    float highest=
        network->activations[output_layer][0];


    // Copy the output probabilities
    for(int i=0;i<10;i++){

        probabilities[i]=
            network->activations[output_layer][i];

        if(!isfinite(probabilities[i])){

            printf(
                "OUTPUT INVALID AT %d:%f\n",
                i,
                probabilities[i]
            );

            return -1;
        }
    }


    // Find the digit with the highest probability
    for(int i=1;i<10;i++){

        if(probabilities[i]>highest){

            highest=probabilities[i];
            prediction=i;
        }
    }


    printf("\n");
    printf("Predicted:%d\n",prediction);

    printf(
        "Confidence:%.2f%%\n",
        highest*100.0f
    );

    printf("PROBABILITIES:\n");

    for(int i=0;i<10;i++){

        printf(
            "%d:%.2f%%\n",
            i,
            probabilities[i]*100.0f
        );
    }

    return prediction;
}


// Draw the predicted digit using seven segments
static void draw_seven_segment(
    SDL_Renderer *renderer,
    int digit,
    int panel_x,
    int panel_y)
{
    int segments[10][7]=
    {
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


    SDL_SetRenderDrawColor(
        renderer,
        255,
        255,
        255,
        255
    );


    // Top
    if(segments[digit][0]){

        SDL_FRect r={
            (float)x,
            (float)y,
            (float)w,
            (float)t
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Top right
    if(segments[digit][1]){

        SDL_FRect r={
            (float)(x+w-t),
            (float)y,
            (float)t,
            (float)(h/2)
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Bottom right
    if(segments[digit][2]){

        SDL_FRect r={
            (float)(x+w-t),
            (float)(y+h/2),
            (float)t,
            (float)(h/2)
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Bottom
    if(segments[digit][3]){

        SDL_FRect r={
            (float)x,
            (float)(y+h-t),
            (float)w,
            (float)t
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Bottom left
    if(segments[digit][4]){

        SDL_FRect r={
            (float)x,
            (float)(y+h/2),
            (float)t,
            (float)(h/2)
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Top left
    if(segments[digit][5]){

        SDL_FRect r={
            (float)x,
            (float)y,
            (float)t,
            (float)(h/2)
        };

        SDL_RenderFillRect(renderer,&r);
    }


    // Middle
    if(segments[digit][6]){

        SDL_FRect r={
            (float)x,
            (float)(y+h/2-t/2),
            (float)w,
            (float)t
        };

        SDL_RenderFillRect(renderer,&r);
    }
}


// Draw the prediction and confidence
static void draw_prediction_panel(
    SDL_Renderer *renderer,
    int prediction,
    float *probabilities)
{
    int panel_x=620;
    int panel_y=80;


    if(prediction==-1)
        return;


    // Draw the prediction box
    SDL_SetRenderDrawColor(
        renderer,
        255,
        255,
        255,
        255
    );

    SDL_FRect box={
        (float)panel_x,
        (float)panel_y,
        250.0f,
        150.0f
    };

    SDL_RenderRect(renderer,&box);


    // Draw the big predicted digit
    draw_seven_segment(
        renderer,
        prediction,
        panel_x,
        panel_y
    );


    float confidence=
        probabilities[prediction];


    if(!isfinite(confidence))
        confidence=0.0f;

    if(confidence<0.0f)
        confidence=0.0f;

    if(confidence>1.0f)
        confidence=1.0f;


    // Confidence background
    SDL_SetRenderDrawColor(
        renderer,
        70,
        70,
        70,
        255
    );

    SDL_FRect background={
        (float)panel_x,
        250.0f,
        250.0f,
        15.0f
    };

    SDL_RenderFillRect(
        renderer,
        &background
    );


    // Confidence bar
    SDL_SetRenderDrawColor(
        renderer,
        0,
        255,
        0,
        255
    );

    float confidence_width=
        confidence*250.0f;

    SDL_FRect bar={
        (float)panel_x,
        250.0f,
        confidence_width,
        15.0f
    };

    SDL_RenderFillRect(
        renderer,
        &bar
    );
}


// Draw probability bars for all digits
static void draw_probability_bars(
    SDL_Renderer *renderer,
    float *probabilities,
    int prediction)
{
    int bars_x=620;
    int bars_y=280;

    int bar_width=250;
    int bar_height=25;


    for(int i=0;i<10;i++){

        int y=bars_y+i*35;


        // Background of the bar
        SDL_SetRenderDrawColor(
            renderer,
            70,
            70,
            70,
            255
        );
        SDL_FRect background={(float)bars_x,(float)y,(float)bar_width,(float)bar_height};
        SDL_RenderFillRect(renderer,&background);
        float probability=
            probabilities[i];
        if(!isfinite(probability))
            probability=0.0f;
        if(probability<0.0f)
            probability=0.0f;
        if(probability>1.0f)
            probability=1.0f;
        int width=(int)(probability*bar_width);
        // Highlight the predicted digit
        if(i==prediction){
            SDL_SetRenderDrawColor(renderer,0,255,0,255);
        }
        else{
            SDL_SetRenderDrawColor(renderer,180,180,180,255);
        }
        SDL_FRect bar={(float)bars_x,(float)y,(float)width,(float)bar_height};
        SDL_RenderFillRect(renderer,&bar);
    }
}
// Draw the main canvas
static void draw_canvas(SDL_Renderer *renderer,float *canvas,float mouse_x,float mouse_y,int drawing)
{
    SDL_SetRenderDrawColor(renderer,25,25,25,255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer,255,255,255,255);
    // Draw every pixel of the canvas
    for(int y=0;y<CANVAS_SIZE;y++){
        for(int x=0;x<CANVAS_SIZE;x++){
            if(canvas[y*CANVAS_SIZE+x]>0.0f){
                SDL_FRect pixel={(float)x,(float)y,1.0f,1.0f};
                SDL_RenderFillRect(renderer,&pixel);
            }
        }
    }
    // Show the brush while drawing
    if(drawing){
        SDL_FRect brush={mouse_x-6,mouse_y-6,12,12};

        SDL_RenderFillRect(
            renderer,
            &brush
        );
    }
}
// Print the 28x28 image in the console
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
int gui_run(float *image,Network *network)
{
    printf("1 Entered gui_run()\n");
    // Initialize SDL
    if(!SDL_Init(SDL_INIT_VIDEO)){
        printf("2 SDL initialization failed: %s\n",SDL_GetError());
        return 0;
    }
    printf("2 SDL initialized successfully!\n");
    SDL_Window *window=NULL;
    SDL_Renderer *renderer=NULL;
    printf("3 Creating window...\n");
    // Create the SDL window and renderer
    if(!SDL_CreateWindowAndRenderer(
        "Digit Recognizer",1000,700,0,&window,&renderer))
    {
        printf("4 Window creation failed: %s\n",SDL_GetError());
        SDL_Quit();
        return 0;
    }
    printf("4 Window created successfully!\n");
    int running=1;
    float mouse_x=0.0f;
    float mouse_y=0.0f;
    int drawing=0;
    float canvas[CANVAS_SIZE*CANVAS_SIZE]={0};
    int prediction=-1;
    int save_number =1;
    float probabilities[10]={0};
    while(running){
        SDL_Event event;
        // Process SDL events
        while(SDL_PollEvent(&event)){
            if(event.type==SDL_EVENT_QUIT){
                running=0;
            }
            // Keyboard input
            if(event.type==SDL_EVENT_KEY_DOWN){
                printf("Key pressed:%d\n",event.key.key);
                // Escape closes the program
                if(event.key.key==SDLK_ESCAPE){
                    running=0;
                }
                // C clears the canvas
                if(event.key.key==SDLK_C){
                    clear_canvas(canvas,image,&prediction,probabilities);
                    printf("Canvas Cleared!\n");
                }
                // R resets only the prediction
                if(event.key.key==SDLK_R){
                    prediction=-1;
                    for(int i=0;i<10;i++)
                        probabilities[i]=0.0f;
                    printf("Prediction reset!\n");
                }
                if(event.key.key == SDLK_S){

                    if(image_is_valid(image)){

                        save_image(image,save_number);

                        save_number++;
                    }else{
                        printf("Nothing to save!\n");
                    }
                }
                // Enter runs the neural network
                if(event.key.key==SDLK_RETURN){
                    printf("Enter click\n");
                    int min_x;
                    int min_y;
                    int max_x;
                    int max_y;
                    find_bounding_box(
                        canvas,
                        &min_x,
                        &min_y,
                        &max_x,
                        &max_y
                    );
                    if(max_x==-1){
                        printf("Canvas Empty!\n");
                    }
                    else{
                        printf(
                            "Bounding box: (%d,%d) -> (%d,%d)\n",
                            min_x,
                            min_y,
                            max_x,
                            max_y
                        );
                        // Convert drawing to MNIST format
                        canvas_to_image(
                            canvas,
                            image
                        );
                        // Show the processed 28x28 image
                        print_image(image);
                        printf("\nRunning neural network...\n");
                        prediction=predict_digit(image,network,probabilities);
                    }
                }
            }
            // Mouse movement
            if(event.type==SDL_EVENT_MOUSE_MOTION){
                mouse_x=event.motion.x;
                mouse_y=event.motion.y;
                if(drawing){
                    int x=(int)mouse_x;
                    int y=(int)mouse_y;
                    int radius=6;
                    // Draw a circular brush
                    for(int dy=-radius;dy<=radius;dy++){
                        for(int dx=-radius;dx<=radius;dx++){
                            int px=x+dx;
                            int py=y+dy;
                            if(px>=0 && px<CANVAS_SIZE && py>=0 && py<CANVAS_SIZE)
                            {
                                if(dx*dx+dy*dy<=radius*radius)
                                {
                                    canvas[
                                        py*CANVAS_SIZE+px
                                    ]=1.0f;
                                }
                            }
                        }
                    }
                }
            }
            // Start drawing
            if(event.type==SDL_EVENT_MOUSE_BUTTON_DOWN){
                if(event.button.button==SDL_BUTTON_LEFT){
                    drawing=1;
                }
            }
            // Stop drawing
            if(event.type==SDL_EVENT_MOUSE_BUTTON_UP){
                if(event.button.button==SDL_BUTTON_LEFT){
                   drawing=0;
                }
            }
        }
        // Draw the canvas
        draw_canvas(
            renderer,
            canvas,
            mouse_x,
            mouse_y,
            drawing
        );
        // Draw prediction panel
        draw_prediction_panel(
            renderer,
            prediction,
            probabilities
        );
        // Draw all probability bars
        draw_probability_bars(
            renderer,
            probabilities,
            prediction
        );
        // Show everything on the screen
        SDL_RenderPresent(renderer);
    }
    printf(
        "5 Closing window...\n"
    );
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
}