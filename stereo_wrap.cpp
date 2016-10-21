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

Mat  getMat(string &filename){
	
	FileStorage fs(filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", filename);
//            return ;
        }

        Mat R;
        fs["R"] >> R;
		fs.release();    
	return R;
      
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

void future_warp(Mat R){
	

	string imagelistfn = "stereo_gray_pro.xml";
	vector<string>goodImageList;
	readStringList(imagelistfn, goodImageList);

	 int i,nimages = (int)goodImageList.size()/2;
	 for(  i = 0; i < nimages; i++ )
	 {
	        	        
	        	Mat src = imread(goodImageList[i*2+1], 1);
//			cout << src.size();
//			cout << R<<endl;
			Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
//			warpAffine(src,warp_dst,R,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
//			warpAffine( src, warp_dst, R, warp_dst.size() );
			warpPerspective(src,warp_dst,R,warp_dst.size(),CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
			char buf[256] = {0};
			sprintf(buf,"data/out/warp_%d.jpg",i);
		        imwrite(buf,warp_dst);					
      	 }
	 
}


int main(){
	Mat R;
	string imagelistfn = "extrinsics.yml";
//	R=getMat(imagelistfn);
//        future_warp(R);

	Point2f srcPoint[4];
	Point2f dstPoint[4];
	Mat warp_mat(3,3,CV_32FC1);

	srcPoint[0] = Point2f(132.8,777.385);
	srcPoint[1] = Point2f(1910.2,768.4);
	srcPoint[2] = Point2f(1357.3,1118.5);
	srcPoint[3] = Point2f(1925.7,1175.6);

	dstPoint[0] = Point2f(1449.4,775.8);
	dstPoint[1] = Point2f(2044.4,767.4);
	dstPoint[2] = Point2f(1461.6,1193);
	dstPoint[3] = Point2f(2043.2,1175);
	
	warp_mat = getPerspectiveTransform(srcPoint,dstPoint);
	
	cout<<"warp_mat"<<warp_mat<<endl;
	 future_warp(warp_mat);

}



