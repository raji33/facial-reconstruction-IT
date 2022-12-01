#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;


Point pt1(-1, -1);
bool pt1New = false;
Point pt2(-1, -1);
bool pt2New = false;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    

    if (event == EVENT_LBUTTONDOWN)
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        pt1.x = x;
        pt1.y = y;
        pt1New = true;
    }
    else if (event == EVENT_RBUTTONDOWN)
    {
        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        pt2.x = x;
        pt2.y = y;
        pt2New = true;
    }
   
}




//**important when running code***
//when image pops up you need to click the dimensions or a ruler.
//user left click one end of the ruler. then click space for the updated point to show
//user right click other end of the ruler and click space to show both points
//After both points show up and have come from a left and right mouse click. click space bar to then continue to run the code
//if no ruler is present in the image then just click two random spots 
//this will not mess up the code it will just give inaccurate measurements when specifiying distance in mm to move features



//This is main method that creates faceDectors and laods landmark detector. 
//This method then runs the image through and generates points for what the ideal features should be. 
int main(int argc, char** argv)
{
    // Load Face Detector
    CascadeClassifier faceDetector("./haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("./lbfmodel.yaml");
    


    // Variable to store a video frame and its grayscale 
    Mat frame, gray;
    frame = imread("./new.jpg");
    double scale_down = 0.5;
    //this is the image for only landmarks being plotted
    Mat scaled_frame;
    //resize
    resize(frame, scaled_frame, Size(), scale_down, scale_down, INTER_LINEAR);

    Mat copy = scaled_frame;

    //this is the image where i will plot the new coordinates of the facial features
    Mat updateFeatures = imread("./new.jpg");
    resize(updateFeatures, updateFeatures, Size(), scale_down, scale_down, INTER_LINEAR);

    //generate image to plot both points on
    Mat containBothPoints = imread("./new.jpg");
    resize(containBothPoints, containBothPoints, Size(), scale_down, scale_down, INTER_LINEAR);


    //find coordinates for the ruler
    bool rulerMarked = false;
    bool rulerHasBeenShown = false;
    while (rulerHasBeenShown == false) {

        //Create a window
        namedWindow("Detection", 1);

        //set the callback function for any mouse event
        setMouseCallback("Detection", CallBackFunc, NULL);


        //show the image for user to select the ruler coordinates
        imshow("Detection", copy);
        cv::waitKey(0);

        if (rulerMarked == false) {
            if (pt1New) {
                circle(copy, pt1, 3, Scalar(255, 200, 0), FILLED);

            }
            if (pt2New) {
                circle(copy, pt2, 3, Scalar(255, 200, 0), FILLED);

            }
        }

        //does this first because if rulerMarked has just been completed it has not yet been shown
        if (rulerMarked) {
            rulerHasBeenShown = true;
        }

        if (pt1New && pt2New) {
            rulerMarked = true;
        }

    }

   

    //check the points and want to determine the length of a ruler 
    //305 mm = 1 ft

    double horizontal = abs(pt1.x - pt2.x);
    double vertical = abs(pt1.y - pt2.y);
    double milliPerPixel;

    //checks which direction (vertical or horizontal) is larger to identify how many pixels for 305 mm
    if (horizontal > vertical) {
        //figure out how many pixels is 1mm
        milliPerPixel = 305 / horizontal;

    }else if (vertical > horizontal) {
        //figure out how many pixels is 1mm
        milliPerPixel = 305 / vertical ;

    }

        

    // This is the link referencing where I found the lbfmodeland how i implemented the facemark detection.
    // Link is : https://chowdera.com/2021/02/20210202135455626M.html
        // Find face
        vector<Rect> faces;
        // Convert frame to grayscale because
        // faceDetector requires grayscale image.
        cv::cvtColor(scaled_frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        faceDetector.detectMultiScale(gray, faces);

        // Variable for landmarks. 
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image.
        //the first array index of this 2-d vector corresponds to face and second arry index corresponds to that facelandmark specific point
        vector< vector<Point2f> > landmarks;


        // Run landmark detector
        //takes in image and the faces it identified and fits landmarks to the face
        bool success = facemark->fit(scaled_frame, faces, landmarks);




        //can tell them how far off their features are from golden ratio  1:1.618
     
        //serves as ratio 1 marker to base all other mesurement ratios -- because it does not change as one ages
        //distance between eyes called intercanthal measurement
        double btwnEyes = abs(landmarks[0][39].x - landmarks[0][42].x);
        cout << "Between eyes is:  " << btwnEyes << endl;
        double widthLEye = abs(landmarks[0][36].x - landmarks[0][39].x);
        double widthLEyeExpected = btwnEyes;
        double diff1 = widthLEye - widthLEyeExpected;
        cout << "Left eye width is:   " << widthLEye << "  expected is : " << widthLEyeExpected << endl;
        if (diff1 > 0) {
            cout << "Left eye width needs to be reduced by " << abs(diff1) * milliPerPixel << " mm" << endl;
            //this gets the width of original eye and subtracts the expected
            double change =  (widthLEye - widthLEyeExpected) / 2;

            //change each position and store it in a point
            Point points[6];
            //coresponds to point 36
            points[0].x = (landmarks[0][36].x + change);
            points[0].y = landmarks[0][36].y;

            //coresponds to point 39
            points[3].x = (landmarks[0][39].x - change);
            points[3].y = landmarks[0][39].y;

            //coresponds to point 37
            points[1].x = (landmarks[0][37].x);
            points[1].y = landmarks[0][37].y;

            //coresponds to point 38
            points[2].x = (landmarks[0][38].x);
            points[2].y = landmarks[0][38].y;

            //coresponds to point 40
            points[4].x = (landmarks[0][40].x);
            points[4].y = landmarks[0][40].y;

            //coresponds to point 41
            points[5].x = (landmarks[0][41].x);
            points[5].y = landmarks[0][41].y;

            
            for (int turn = 0; turn < 6; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);
            }

        }
        else if (diff1 < 0) {
            cout << "Left eye width needs to be increased by " << abs(diff1) * milliPerPixel << " mm" << endl;

            //this gets the expected minus the original length. Divide by 2. This gets the change needed for each point.
            double change = (widthLEyeExpected - widthLEye) / 2;
           

            //change each position and store it in a point
            Point points[6];
            //coresponds to point 36
            points[0].x = (landmarks[0][36].x - change);
            points[0].y = landmarks[0][36].y;

            //coresponds to point 39
            points[3].x = (landmarks[0][39].x + change);
            points[3].y = landmarks[0][39].y;

            //coresponds to point 37
            points[1].x = (landmarks[0][37].x);
            points[1].y = landmarks[0][37].y;

            //coresponds to point 38
            points[2].x = (landmarks[0][38].x);
            points[2].y = landmarks[0][38].y;

            //coresponds to point 40
            points[4].x = (landmarks[0][40].x);
            points[4].y = landmarks[0][40].y;

            //coresponds to point 41
            points[5].x = (landmarks[0][41].x);
            points[5].y = landmarks[0][41].y;

          


            for (int turn = 0; turn < 6; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else {
            cout << "No changes need to be made" << endl;

        }

        double widthREye = abs(landmarks[0][42].x - landmarks[0][45].x);
        double widthREyeExpected = btwnEyes;
        double diff2 = widthREye - widthREyeExpected;

        cout << "Right eye width is:   " << widthREye << "  expected is : " << widthREyeExpected << endl;
        if (diff2 > 0) {
            cout << "Right eye width needs to be reduced by " << abs(diff2) * milliPerPixel << " mm" << endl;
            double change = (widthREye - widthREyeExpected) / 2;

            //change each position and store it in a point
            Point points[6];
            //coresponds to point 42
            points[0].x = (landmarks[0][42].x + change);
            points[0].y = landmarks[0][42].y;

            //coresponds to point 43
            points[1].x = (landmarks[0][43].x);
            points[1].y = landmarks[0][43].y;

            //coresponds to point 44
            points[2].x = (landmarks[0][44].x);
            points[2].y = landmarks[0][44].y;

            //coresponds to point 45
            points[3].x = (landmarks[0][45].x - change);
            points[3].y = landmarks[0][45].y;

            //coresponds to point 46
            points[4].x = (landmarks[0][46].x);
            points[4].y = landmarks[0][46].y;

            //coresponds to point 47
            points[5].x = (landmarks[0][47].x);
            points[5].y = landmarks[0][47].y;


            for (int turn = 0; turn < 6; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff2 < 0) {
            cout << "Right eye width needs to be increased by " << abs(diff2) * milliPerPixel << " mm" << endl;
            double change = (widthREyeExpected - widthREye) / 2;

            //change each position and store it in a point
            Point points[6];
            //coresponds to point 42
            points[0].x = (landmarks[0][42].x - change);
            points[0].y = landmarks[0][42].y;

            //coresponds to point 43
            points[1].x = (landmarks[0][43].x);
            points[1].y = landmarks[0][43].y;

            //coresponds to point 44
            points[2].x = (landmarks[0][44].x);
            points[2].y = landmarks[0][44].y;

            //coresponds to point 45
            points[3].x = (landmarks[0][45].x + change);
            points[3].y = landmarks[0][45].y;

            //coresponds to point 46
            points[4].x = (landmarks[0][46].x);
            points[4].y = landmarks[0][46].y;

            //coresponds to point 47
            points[5].x = (landmarks[0][47].x);
            points[5].y = landmarks[0][47].y;


            for (int turn = 0; turn < 6; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else {
            cout << "No changes need to be made" << endl;

        }

        //acts as ratio 1
        //the act is from the inner point of eyebrow to the point before it curls
        double eyebrowLArch = abs(landmarks[0][19].x - landmarks[0][21].x);
        double eyebrowRArch = abs(landmarks[0][22].x - landmarks[0][24].x);
        double eyebrowArchExpected = eyebrowLArch;
        //double diff5 = abs(eyebrowLArch - eyebrowArchExpected);
        double diff6 = eyebrowRArch - eyebrowArchExpected;

        cout << "Width of left eyebrow arch is:   " << eyebrowLArch << endl;

        cout << "Width of right eyebrown arch is:   " << eyebrowLArch << "  expected is : " << eyebrowArchExpected << endl;
        if (diff6 > 0) {
            cout << "Width of right eyebrow arch needs to be reduced by " << abs(diff6) * milliPerPixel << " mm" << endl;

        }
        else if (diff6 < 0) {
            cout << "Width of right eyebrow arch needs to be reduced by " << abs(diff6) * milliPerPixel << " mm" << endl;

        }
        else {
            cout << "No changes need to be made" << endl;

        }


        //this value is ratio 1.618 compared to eyebrow Arch
        double eyebrowFullL = abs(landmarks[0][17].x - landmarks[0][21].x);
        double eyebrowFullR = abs(landmarks[0][22].x - landmarks[0][26].x);
        double eyebrowFullArchExpected = eyebrowArchExpected * 1.618;
        double diff7 = eyebrowFullL - eyebrowFullArchExpected;
        double diff8 = eyebrowFullR - eyebrowFullArchExpected;


        cout << "Width of full left eyebrown is:   " << eyebrowFullL << "  expected is : " << eyebrowFullArchExpected << endl;
        if (diff7 > 0) {
            cout << "Width of left full eyebrow needs to be reduced by " << abs(diff7) * milliPerPixel << " mm" << endl;

            //gives the amount of pixels to move the points
            double change = (eyebrowFullL - eyebrowFullArchExpected) / 2;

            //change each position and store it in a point
            Point points[5];
            //coresponds to point 17
            points[0].x = (landmarks[0][17].x + change);
            points[0].y = landmarks[0][17].y;

            //coresponds to point 18
            points[1].x = (landmarks[0][18].x + change);
            points[1].y = landmarks[0][18].y;

            //coresponds to point 19 -- do not change bc its the middle point
            points[2].x = (landmarks[0][19].x);
            points[2].y = landmarks[0][19].y;

            //coresponds to point 20
            points[3].x = (landmarks[0][20].x - change);
            points[3].y = landmarks[0][20].y;

            //coresponds to point 21
            points[4].x = (landmarks[0][21].x - change);
            points[4].y = landmarks[0][21].y;


            for (int turn = 0; turn < 5; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff7 < 0) {
            cout << "Width of left full eyebrow needs to be increased by " << abs(diff7) * milliPerPixel << " mm" << endl;
            //gives the amount of pixels to move the points
            double change = (eyebrowFullArchExpected - eyebrowFullL) / 2;

            //change each position and store it in a point
            Point points[5];
            //coresponds to point 17
            points[0].x = (landmarks[0][17].x - change);
            points[0].y = landmarks[0][17].y;

            //coresponds to point 18
            points[1].x = (landmarks[0][18].x - change);
            points[1].y = landmarks[0][18].y;

            //coresponds to point 19 -- do not change bc its the middle point
            points[2].x = (landmarks[0][19].x);
            points[2].y = landmarks[0][19].y;

            //coresponds to point 20
            points[3].x = (landmarks[0][20].x + change);
            points[3].y = landmarks[0][20].y;

            //coresponds to point 21
            points[4].x = (landmarks[0][21].x + change);
            points[4].y = landmarks[0][21].y;


            for (int turn = 0; turn < 5; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }
        }
        else {
            cout << "No changes need to be made" << endl;

        }

        cout << "Width of full right eyebrown is:   " << eyebrowFullR << "  expected is : " << eyebrowFullArchExpected << endl;

        if (diff8 > 0) {
            cout << "Width of right full eyebrow needs to be reduced by " << abs(diff8) * milliPerPixel << " mm" << endl;
            //gives the amount of pixels to move the points
            double change = (eyebrowFullR - eyebrowFullArchExpected) / 2;

            //change each position and store it in a point
            Point points[5];
            //coresponds to point 22
            points[0].x = (landmarks[0][22].x + change);
            points[0].y = landmarks[0][22].y;

            //coresponds to point 23
            points[1].x = (landmarks[0][23].x + change);
            points[1].y = landmarks[0][23].y;

            //coresponds to point 24 -- do not change bc its the middle point
            points[2].x = (landmarks[0][24].x);
            points[2].y = landmarks[0][24].y;

            //coresponds to point 25
            points[3].x = (landmarks[0][25].x - change);
            points[3].y = landmarks[0][25].y;

            //coresponds to point 26
            points[4].x = (landmarks[0][26].x - change);
            points[4].y = landmarks[0][26].y;

            //plots points on image
            for (int turn = 0; turn < 5; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff8 < 0) {
            cout << "Width of right full eyebrow needs to be increased by " << abs(diff8) * milliPerPixel << " mm" << endl;
            double change = (eyebrowFullArchExpected - eyebrowFullR) / 2;

            //change each position and store it in a point
            Point points[5];
            //coresponds to point 22
            points[0].x = (landmarks[0][22].x - change);
            points[0].y = landmarks[0][22].y;

            //coresponds to point 23
            points[1].x = (landmarks[0][23].x - change);
            points[1].y = landmarks[0][23].y;

            //coresponds to point 24 -- do not change bc its the middle point
            points[2].x = (landmarks[0][24].x);
            points[2].y = landmarks[0][24].y;

            //coresponds to point 25
            points[3].x = (landmarks[0][25].x + change);
            points[3].y = landmarks[0][25].y;

            //coresponds to point 26
            points[4].x = (landmarks[0][26].x + change);
            points[4].y = landmarks[0][26].y;

            //plots points on image
            for (int turn = 0; turn < 5; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }
        }
        else {
            cout << "No changes need to be made" << endl;

        }


        //checks the nose length and compares it to what it should be
        double noseLength = abs(landmarks[0][27].y - landmarks[0][33].y);
        double noseLengthExpected = eyebrowArchExpected * 1.618;
        double diff9 = noseLength - noseLengthExpected;

        Point nosePoints[9];
        //this will combine the points of the nose width and length to alter then accordingly. As altering the length and then altering the width result in two seperate plots so we need to combine them.
        //index goes from index 0 = point 27
        //index 8 = point 35
        //initialize points
        
        


        cout << "Nose length is:   " << noseLength << "  expected is : " << noseLengthExpected << endl;
        if (diff9 > 0) {
            cout << "Length of nose needs to be reduced by " << abs(diff9) * milliPerPixel << " mm" << endl;
            double change = (noseLength - noseLengthExpected) / 2;

            //changes the coordinates to reflect the reduced nose length 
            // do not change the x coordinate as length is in vertical axis
            nosePoints[0].x = landmarks[0][27].x;
            nosePoints[0].y = landmarks[0][27].y + change;
            nosePoints[1].x = landmarks[0][28].x;
            nosePoints[1].y = landmarks[0][28].y + change;
            nosePoints[2].x = landmarks[0][29].x;
            //do not change point 29 y coordinate bc its the middle point and moving all other points closer towards it
            nosePoints[2].y = landmarks[0][29].y;
            nosePoints[3].x = landmarks[0][30].x;
            nosePoints[3].y = landmarks[0][30].y - change ;
            nosePoints[4].x = landmarks[0][31].x;
            nosePoints[4].y = landmarks[0][31].y - change;
            nosePoints[5].x = landmarks[0][32].x;
            nosePoints[5].y = landmarks[0][32].y - change;
            nosePoints[6].x = landmarks[0][33].x;
            nosePoints[6].y = landmarks[0][33].y - change;
            nosePoints[7].x = landmarks[0][34].x;
            nosePoints[7].y = landmarks[0][34].y - change;
            nosePoints[8].x = landmarks[0][35].x;
            nosePoints[8].y = landmarks[0][35].y - change;

        }
        else if (diff9 < 0) {
            cout << "Length of nose needs to be increased by " << abs(diff9) * milliPerPixel << " mm" << endl;
            double change = (noseLengthExpected - noseLength) / 2;

            //changes the coordinates to reflect the increased nose length 
           // do not change the x coordinate as length is in vertical axis
            nosePoints[0].x = landmarks[0][27].x;
            nosePoints[0].y = landmarks[0][27].y - change;
            nosePoints[1].x = landmarks[0][28].x;
            nosePoints[1].y = landmarks[0][28].y - change;
            nosePoints[2].x = landmarks[0][29].x;
            //do not change point 29 y coordinate bc its the middle point and moving all other points closer towards it
            nosePoints[2].y = landmarks[0][29].y;
            nosePoints[3].x = landmarks[0][30].x;
            nosePoints[3].y = landmarks[0][30].y + change;
            nosePoints[4].x = landmarks[0][31].x;
            nosePoints[4].y = landmarks[0][31].y + change;
            nosePoints[5].x = landmarks[0][32].x;
            nosePoints[5].y = landmarks[0][32].y + change;
            nosePoints[6].x = landmarks[0][33].x;
            nosePoints[6].y = landmarks[0][33].y + change;
            nosePoints[7].x = landmarks[0][34].x;
            nosePoints[7].y = landmarks[0][34].y + change;
            nosePoints[8].x = landmarks[0][35].x;
            nosePoints[8].y = landmarks[0][35].y + change;

        }
        else {
            cout << "No changes need to be made" << endl;

        }

        //this is width of mouth and acts as ratio 1.618
        double mouthWidth = abs(landmarks[0][48].x - landmarks[0][54].x);
        //double mouthWidthExpected = noseWidth * 1.618;
        cout << "Width of mouth is:   " << mouthWidth <<  endl;

        //this is ratio 1 and using mouthWidth ratio we find the expected nose width 
        double noseWidth = abs(landmarks[0][31].x - landmarks[0][35].x);
        double noseWidthExpected = mouthWidth / 1.618;
        double diff10 = noseWidth - noseWidthExpected;

        cout << "Nose width is:   " << noseWidth << "  expected is : " << noseWidthExpected << endl;

        //in these methods will change the nosePoints x coordinates to adjust for the nose width size
        if (diff10 > 0) {
            cout << "Nose width needs to be reduced by " << abs(diff10) * milliPerPixel << " mm" << endl;
            double change = (noseWidth - noseWidthExpected) / 2;
            //changes only points 31-35 
            //nosePoints index is [4] - [8]
            nosePoints[4].x = landmarks[0][31].x + change;
            nosePoints[5].x = landmarks[0][32].x + change;
            //do not change point 6 because its middle point and I am moving all the other points closer to it
            nosePoints[6].x = landmarks[0][33].x;
            nosePoints[7].x = landmarks[0][34].x - change;
            nosePoints[8].x = landmarks[0][35].x - change;
            for (int turn = 0; turn < 9; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, nosePoints[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, nosePoints[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff10 < 0) {
            cout << "Nose width needs to be increased by " << abs(diff10) * milliPerPixel << " mm" << endl;
            double change = (noseWidthExpected - noseWidth) / 2;
            //changes only points 31-35 
            //nosePoints index is [4] - [8]
            nosePoints[4].x = landmarks[0][31].x - change;
            nosePoints[5].x = landmarks[0][32].x - change;
            //do not change point 6 because its middle point and I am moving all the other points futher from it
            nosePoints[6].x = landmarks[0][33].x;
            nosePoints[7].x = landmarks[0][34].x + change;
            nosePoints[8].x = landmarks[0][35].x + change;

            for (int turn = 0; turn < 9; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, nosePoints[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, nosePoints[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else {
            cout << "No changes need to be made" << endl;

        }

        //vertical checks
        //acts as the 1.618 ratio
        //the height nose to top of mouth is based on the changed nose length from above
        //this is done so we arent shortening the nose length and then elongating chin based on old length 
        double heightNoseToTopMouth = abs(nosePoints[0].y - landmarks[0][62].y);

        cout << "Height of nose to bottom of top lip is:   " << heightNoseToTopMouth << endl;

        //tells you if your chin needs to be lifted up or extended more down
        double heightChinToTopBottomLip = abs(landmarks[0][66].y - landmarks[0][8].y);
        double heightChinToTopBottomLipExpected = heightNoseToTopMouth / 1.618;
        double diff11 = heightChinToTopBottomLip - heightChinToTopBottomLipExpected;

        cout << "Height from chin to top of bottom lip is:   " << heightChinToTopBottomLip << "  expected is : " << heightChinToTopBottomLipExpected << endl;

        //initialize it here so in the width of face check we can use points to draw new jawline
        //this doesnt matter as I commented out the width checks because i had no way of changing the rest of the jawline points to match the new face width 
        Point pointz[3];

        if (diff11 > 0) {
            cout << "Chin length needs to be reduced by " << abs(diff11) * milliPerPixel << " mm" << endl;
            double change = (heightChinToTopBottomLip - heightChinToTopBottomLipExpected);
            pointz[0].x = landmarks[0][7].x;
            pointz[0].y = landmarks[0][7].y - change;
            pointz[1].x = landmarks[0][8].x;
            pointz[1].y = landmarks[0][8].y - change;
            pointz[2].x = landmarks[0][9].x;
            pointz[2].y = landmarks[0][9].y - change;
            for (int turn = 0; turn < 3; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, pointz[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, pointz[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff11 < 0) {
            cout << "Chin length needs to be increased by " << abs(diff11) * milliPerPixel << " mm" << endl;
            double change = (heightChinToTopBottomLipExpected - heightChinToTopBottomLip);           
            pointz[0].x = landmarks[0][7].x;
            pointz[0].y = landmarks[0][7].y + change;
            pointz[1].x = landmarks[0][8].x;
            pointz[1].y = landmarks[0][8].y + change;
            pointz[2].x = landmarks[0][9].x;
            pointz[2].y = landmarks[0][9].y + change;
            for (int turn = 0; turn < 3; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, pointz[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, pointz[turn], 3, Scalar(15, 15, 255), FILLED);

            }
        }
        else {
            cout << "No changes need to be made" << endl;

        }



        //***** Commented out the checks for the width of both sides of face
        //this was done because I was unsure if the actual mesurement was from eye corner to the side of the jawline or to the highpoint of the cheekbone (which is not identified using the current model)
        // also could not figure out what type of equation (curve) was needed to find x coordinates of the jawline points given the altered point

        //double widthLFace = abs(landmarks[0][0].x - landmarks[0][39].x);
        //double widthLFaceExpected = btwnEyes * 1.618;
        //double diff3 = widthLFace - widthLFaceExpected;

        //cout << "Width of left face is:   " << widthLFace << "  expected is : " << widthLFaceExpected << endl;
        //if (diff3 > 0) {
        //    cout << "Width of left face needs to be reduced by " << abs(diff3) * milliPerPixel << " mm" << endl;

        //    Point points[7];

        //    //[0] represents point 0 on facemap
        //    points[0].x = landmarks[0][0].x + (abs(diff3));
        //    points[0].y = landmarks[0][0].y;

        //    //grab x distance between new left side of face and new chin point at point 7
        //   //pointz[0] corresponds to new point 7
        //    double distance = abs(points[0].x - pointz[0].x);

        //    //divide distance by 7 because there are 7 "gaps" between those 7 points
        //    double change = distance / 7;

        //    cout << "Distance per face movement is " << change << endl;

        //    cout << "Point 0 is at : " << points[0].x << endl;
        //    points[1].x = points[0].x + change;
        //    points[1].y = landmarks[0][1].y;
        //    cout << "Point 1 is at : " << points[1].x << endl;

        //    points[2].x = points[1].x + change;
        //    points[2].y = landmarks[0][2].y;
        //    cout << "Point 2 is at : " << points[2].x << endl;

        //    points[3].x = points[2].x + change;
        //    points[3].y = landmarks[0][3].y;
        //    cout << "Point 3 is at : " << points[3].x << endl;

        //    points[4].x = points[3].x + change;
        //    points[4].y = landmarks[0][4].y;
        //    cout << "Point 4 is at : " << points[4].x << endl;

        //    points[5].x = points[4].x + change;
        //    points[5].y = landmarks[0][5].y;
        //    cout << "Point 5 is at : " << points[5].x << endl;

        //    points[6].x = points[5].x + change;
        //    points[6].y = landmarks[0][6].y;
        //    cout << "Point 6 is at : " << points[6].x << endl;
        //    
        //    for (int turn = 0; turn < 7; turn++) {
        //        //sets color of circle to be red
        //        circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
        //    }

        //}
        //else if (diff3 < 0) {
        //    cout << "Width of left face needs to be increased by " << abs(diff3) * milliPerPixel << " mm" << endl;
        //    Point points[7];
        //    points[0].x = landmarks[0][0].x - (abs(diff3));
        //    points[0].y = landmarks[0][0].y;

        //    //grab x distance between new left side of face and new chin point at point 7
        //    //pointz[0] corresponds to new point 7
        //    double distance = abs(points[0].x - pointz[0].x);
        //   
        //    //divide distance by 7 because there are 7 "gaps" between those 7 points
        //    double change = distance / 7;

        //    cout << "Distance per face movement is " << change << endl;

        //    cout << "Point 0 is at : " << points[0].x << endl;
        //    points[1].x = points[0].x + change;
        //    points[1].y = landmarks[0][1].y;
        //    cout << "Point 1 is at : " << points[1].x << endl;

        //    points[2].x = points[1].x + change;
        //    points[2].y = landmarks[0][2].y;
        //    cout << "Point 2 is at : " << points[2].x << endl;

        //    points[3].x = points[2].x + change;
        //    points[3].y = landmarks[0][3].y;
        //    cout << "Point 3 is at : " << points[3].x << endl;

        //    points[4].x = points[3].x + change;
        //    points[4].y = landmarks[0][4].y;
        //    cout << "Point 4 is at : " << points[4].x << endl;

        //    points[5].x = points[4].x + change;
        //    points[5].y = landmarks[0][5].y;
        //    cout << "Point 5 is at : " << points[5].x << endl;

        //    points[6].x = points[5].x + change;
        //    points[6].y = landmarks[0][6].y;
        //    cout << "Point 6 is at : " << points[6].x << endl;

        //    for (int turn = 0; turn < 7; turn++) {
        //        //sets color of circle to be red
        //        circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
        //        //imshow("", updateFeatures);
        //        //cv::waitKey(0);
        //    }
        //}

        //else {
        //    cout << "No changes need to be made" << endl;

        //}


        //double widthRFace = abs(landmarks[0][42].x - landmarks[0][16].x);
        //double widthRFaceExpected = btwnEyes * 1.618;
        //double diff4 = widthRFace - widthRFaceExpected;
        //cout << "Width of right face is:   " << widthRFace << "  expected is : " << widthRFaceExpected << endl;
        //if (diff4 > 0) {
        //    cout << "Width of right face needs to be reduced by " << abs(diff4) * milliPerPixel << " mm" << endl;
        //    Point points[1];

        //    points[0].x = landmarks[0][16].x - abs(diff4);
        //    points[0].y = landmarks[0][0].y;
        //    for (int turn = 0; turn < 1; turn++) {
        //        //sets color of circle to be red
        //        circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
        //    }
        //}
        //else if (diff4 < 0) {
        //    cout << "Width of right face needs to be increased by " << abs(diff4) * milliPerPixel << " mm" << endl;
        //    Point points[1];

        //    points[0].x = landmarks[0][016].x + abs(diff4);
        //    points[0].y = landmarks[0][0].y;
        //    for (int turn = 0; turn < 1; turn++) {
        //        //sets color of circle to be red
        //        circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
        //    }
        //}
        //else {
        //    cout << "No changes need to be made" << endl;

        //}

        //acts as ratio 1
        double topLipThickness = abs(landmarks[0][51].y - landmarks[0][62].y);
        cout << "Top lip thickness is:   " << topLipThickness << endl;

        //plot top lip coordinates for both image plots as well
        Point lipsFeature[17];
        //index[0] = point 48
        //index[16] = point 64

        //this loop will initialize all the values to what they should be
        for (int number = 0; number < 17; number++) {
            //initializes point
            lipsFeature[number].x = landmarks[0][48 + number].x;
            lipsFeature[number].y = landmarks[0][48 + number].y;

            //plots point on both plots
            circle(updateFeatures, lipsFeature[number], 3, Scalar(15, 15, 255), FILLED);
            circle(containBothPoints, lipsFeature[number], 3, Scalar(15, 15, 255), FILLED);

            

        }

        //bottom lip thickness acts as ratio 1.618 and using top lip thicness we find expected bottom lip thickness
        double bottomLipThickness = abs(landmarks[0][66].y - landmarks[0][57].y);
        double bottomLipThicknessExpected = topLipThickness * 1.618; 
        double diff12 = bottomLipThickness - bottomLipThicknessExpected;


        cout << "Bottom Lip thickness is:   " << bottomLipThickness << "  expected is : " << bottomLipThicknessExpected << endl;
        if (diff12 > 0) {
            cout << "Bottom lip thickness needs to be reduced by " << abs(diff12) * milliPerPixel << " mm" << endl;
            double change = (bottomLipThickness - bottomLipThicknessExpected) / 2;
            Point points[3];
            //will change the y coordinates only because the lip thickness is in vertical direction
            //reduces the y coordinates by specified amount because we are reducing the thickness
            //coresponds to point 67
            points[0].x = (landmarks[0][67].x);
            points[0].y = landmarks[0][67].y - change;

            //coresponds to point 66
            points[1].x = (landmarks[0][66].x);
            points[1].y = landmarks[0][66].y - change;

            //coresponds to point 65
            points[2].x = (landmarks[0][65].x);
            points[2].y = landmarks[0][65].y -  change;
            for (int turn = 0; turn < 3; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else if (diff12 < 0) {
            cout << "Bottom lip thickness needs to be increased by " << abs(diff12) * milliPerPixel << " mm" << endl;
            double change = (bottomLipThicknessExpected - bottomLipThickness) / 2;
            Point points[3];
            //will change the y coordinates only because the lip thickness is in vertical direction
            //increased the y coordinates by specified amount because we are increasing the thickness
            //coresponds to point 67
            points[0].x = (landmarks[0][67].x);
            points[0].y = landmarks[0][67].y + change;

            //coresponds to point 66
            points[1].x = (landmarks[0][66].x);
            points[1].y = landmarks[0][66].y + change;

            //coresponds to point 65
            points[2].x = (landmarks[0][65].x);
            points[2].y = landmarks[0][65].y + change;
            for (int turn = 0; turn < 3; turn++) {
                //sets color of circle to be red
                circle(updateFeatures, points[turn], 3, Scalar(15, 15, 255), FILLED);
                circle(containBothPoints, points[turn], 3, Scalar(15, 15, 255), FILLED);

            }

        }
        else {
            cout << "No changes need to be made" << endl;

        }

        //if the landmark mapping is succesful, plot it on images.  
        if (success)
        {
            // If successful, render the landmarks on the face
            //each landmark index represents the dotss for each serpate face detected
            //landmarks.size() tells us how many faces were detected in image
            //for our program only detecting one face so landmarks.size has to be 1
           for (int i = 0; i < landmarks.size(); i++)
            {
                
                       //plots the landmarks for the face on both images - the scaled_frame and the containBothPoints image
                    drawLandmarks(scaled_frame, landmarks[i]);
                    drawLandmarks(containBothPoints, landmarks[i]);
                
            }

         
        }

        // Display results 
        imshow("Facial Landmark Detection", scaled_frame);
        cv::waitKey(0);


        //display the updated features 
        imshow("Updated Features", updateFeatures);
        cv::waitKey(0);

        //display both plots overlapped
        imshow("Face with both plots", containBothPoints);
        cv::waitKey(0);


}








    
   



