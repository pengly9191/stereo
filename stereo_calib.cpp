/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     Issue tracker: http://code.opencv.org
     GitHub:        https://github.com/Itseez/opencv/
   ************************************************** */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;



static bool readStringList( const string& filename, vector<string>& l );
void future_color_sterecalib(bool useCalibrated ,Mat rmap[][2],Rect validRoi[]);
void future_gray_sterecalib(bool useCalibrated,Mat rmap[][2],Rect validRoi[]);

void future_warp(Mat R);

static void
StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=false, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    bool displayCorners = false;//true;
    const int maxScale = 2;
    const float squareSize = 20.f;  // Set this to your actual square size
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
		
                found = findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
		imwrite("corner.jpg",cimg);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);                         
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,             
                    CALIB_USE_INTRINSIC_GUESS+
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

    cout << "cameraMatrix[0]"<<cameraMatrix[0] <<  endl;
	

    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
//            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
	  
        }
		
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " <<  err/npoints << endl;

     cout << "cameraMatrix[0]"<<cameraMatrix[0] <<  endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);

    cout << "validRoi[0]:"<< validRoi[0]<<endl;

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }

    else
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2,R11,R22;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R11 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R22 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];	
		
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

//    future_warp(R);
    future_gray_sterecalib( false,rmap,validRoi);
}


void future_warp(Mat R){

	string imagelistfn = "stereo_gray_pro.xml";
	vector<string>goodImageList;
	readStringList(imagelistfn, goodImageList);

	 int i,nimages = (int)goodImageList.size()/2;
	 for(  i = 0; i < nimages; i++ )
	 {
	        
	        
	        	Mat src = imread(goodImageList[i*2+1], 0);
			Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
//			warpAffine(src,warp_dst,R,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
			warpAffine( src, warp_dst, R, warp_dst.size() );
			char buf[256] = {0};
			sprintf(buf,"data/out/warp_%d.jpg",i);
		        imwrite(buf,warp_dst);					
      			
	        		
	 }
	 
}



void future_color_sterecalib(bool useCalibrated,Mat rmap[][2],Rect validRoi[]){

    cout << "enter future_sterecalib!!!!!!!!!!!!!"<<endl;
    Mat canvas;
    double sf = 1.0;
  
    int w, h;
    Size imageSize;
    bool isVerticalStereo = false;

    string imagelistfn = "stereo_gray_pro.xml";  
    vector<string> goodImageList;
    readStringList(imagelistfn, goodImageList);
  
    int i, j, k, nimages = (int)goodImageList.size()/2;

    const string& filename = goodImageList[0];
    Mat img = imread(filename,0);
    imageSize = img.size();
	
    w = cvRound(imageSize.width);
    h = cvRound(imageSize.height);
    canvas.create(h, w*2, CV_8UC3);
  
    for(  i = 0; i < nimages; i++ )
    {
        for(  k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
	
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);   
		
	    if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
			
        }
	
#if 1
        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
 #endif
 
	char buf1[256] ={0};
	if (i % 3 == 0)
        	sprintf(buf1,"/home/ply/stereo/data/out/Red-canvas-%d.jpg",(int) (i/3));
	else if (i % 3 ==1)
		sprintf(buf1,"/home/ply/stereo/data/out/Green-canvas-%d.jpg",(int) (i/3));
	else 
		sprintf(buf1,"/home/ply/stereo/data/out/Blue-canvas-%d.jpg",(int) (i/3));		
	imwrite(buf1,canvas);  

	
       
    }
	cout <<" finish!!!!!!!!!!!!!!!!"<<endl;
}


void future_gray_sterecalib(bool useCalibrated,Mat rmap[][2],Rect validRoi[]){

    cout << "enter future_sterecalib!!!!!!!!!!!!!"<<endl;
    Mat canvas;
    double sf = 1.0;
  
    int w, h;
    Size imageSize;
    bool isVerticalStereo = false;

    string imagelistfn = "stereo_gray_pro.xml";  
    vector<string> goodImageList;
    readStringList(imagelistfn, goodImageList);
  
    int i, k, nimages = (int)goodImageList.size()/2;

    const string& filename = goodImageList[0];
    Mat img = imread(filename,0);
    imageSize = img.size();
	
    w = cvRound(imageSize.width);
    h = cvRound(imageSize.height);
    canvas.create(h, w*2, CV_8UC3);
  
    for(  i = 0; i < nimages; i++ )
    {
        for(  k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
	
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);   
		
	    if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
			
        }
	
#if 0
        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 160)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 160 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
 #endif
 
	
	char buf[256] = {0};
	sprintf(buf,"data/out/%d.jpg",i);
        imwrite(buf,canvas);
	imwrite("out.jpg",canvas);	
       
    }
	cout <<" finish!!!!!!!!!!!!!!!!"<<endl;
}



int toCalibXmlFile(int n ){
	
        FileStorage fs("stereo_calib1.xml", FileStorage::WRITE);  
	string PATH = "/home/ply/stereo/data/in/";
        string jpg = ".jpg";
        fs << "imagelist"<<"[";          
        for( int i = 0; i < n; i++ )  
        {             
            string n = static_cast<ostringstream *>( &(ostringstream() << i))->str();
	    string leftname = PATH+"left-pic-"+ n + jpg;
	    string rightname = PATH+"right-pic-"+ n + jpg;
		
	    fs << leftname<<rightname;	    
        }   
	fs << "]";
        fs.release();  
        return 0;  

}

int toColorProXmlFile(int n){

        FileStorage fs("stereo_pro.xml", FileStorage::WRITE);  
	string PATH = "/home/ply/stereo/data/flit/";	
        string jpg = ".jpg";
        fs << "imagelist"<<"[";          
        for( int i = 0; i < n; i++ )  
        {             
            string n = static_cast<ostringstream *>( &(ostringstream() << i))->str();
	    string Blueleftname = PATH+"Blue-left-dst"+ n + jpg;
	    string Bluerightname = PATH+"Blue-right-dst"+ n + jpg;
	    string Greenleftname = PATH+"Green-left-dst"+ n + jpg;
	    string Greenrightname = PATH+"Green-right-dst"+ n + jpg;
	    string Redleftname = PATH+"Red-left-dst"+ n + jpg;
	    string Redrightname = PATH+"Red-right-dst"+ n + jpg;
		
	    fs << Blueleftname<<Bluerightname<<Greenleftname<<Greenrightname<<Redleftname<<Redrightname;	    
        }   
	fs << "]";
        fs.release();  
        return 0;  


}

int toGrayProXmlFile(int n){


        FileStorage fs("stereo_gray_pro.xml", FileStorage::WRITE);  
	string PATH = "/home/ply/stereo/data/pic/";
        string jpg = ".jpg";
        fs << "imagelist"<<"[";          
        for( int i = 0; i < n; i++ )  
        {             
            string n = static_cast<ostringstream *>( &(ostringstream() << i))->str();
	    string leftname = PATH+"left-pic-"+ n + jpg;
	    string rightname = PATH+"right-pic-"+ n + jpg;
		
	    fs << leftname<<rightname;	    
        }   
	fs << "]";
        fs.release();  
        return 0;  

}

static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified = true;

#if 0
    toCalibXmlFile(15);
    toColorProXmlFile(2);
    toGrayProXmlFile(12);
	
#endif 

    imagelistfn = "stereo_calib1.xml";
    boardSize = Size(11, 8);
    vector<string> imagelist;
    readStringList(imagelistfn, imagelist);
  
    StereoCalib(imagelist, boardSize, false, showRectified);
    return 0;
}
