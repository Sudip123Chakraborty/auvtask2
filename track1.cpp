#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include<iostream>
#include </home/sudip/Desktop/winterworkshop/eigen-eigen-323c052e1731/Eigen/Dense>
#include </home/sudip/Desktop/winterworkshop/eigen-eigen-323c052e1731/Eigen/Core>


using namespace cv;
using namespace std;
using namespace Eigen;
Mat img,frame,src,RGB;
double ticks=0;
double x_pr[488*648],y_pr[488*648];


void contour(Mat image);

int main()
{
  int i,j;
  VideoCapture Video("auv.avi");// capturing the video
  //Mat frame;
  Mat bw,hsv;
  while(1)
  {

      Video>>src;
	//cout<<"sudip_chakraborty: "<<RGB.rows<<" "<<RGB.cols<<endl;
      Mat binary(RGB.rows,RGB.cols,CV_8UC1,Scalar(0));
      Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	//cout<<"chakraborty_sudip: "<<src.rows<<" "<<src.cols<<endl;



      Mat fullImageHSV;
      cvtColor(src,RGB, CV_BGR2RGB);

      cvtColor(RGB,bw, CV_RGB2GRAY);//changing rgb to grayscale

      cvtColor(RGB,hsv, CV_RGB2HSV);

      threshold(bw,binary,10,255, THRESH_BINARY_INV);

      for(i=0;i<RGB.rows;i++)
      {
          for(j=0;j<RGB.cols;j++)
          {
            if((bw.at<uchar>(i,j)<40))
            binary.at<uchar>(i,j)=0;

            else binary.at<uchar>(i,j)=255;
          }
      }
/*********ERROSION for smoothing the LINES && DILATION for joining the holes*******************/
        erode(binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );//errosion for clearing noise 
        dilate(binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );//& diletioin for joining small gaps

         
        dilate( binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
        erode(binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );




     // namedWindow("source",WINDOW_NORMAL);
     // namedWindow("RGB image",WINDOW_NORMAL);
     // namedWindow("HSV image",WINDOW_NORMAL);
     // namedWindow("Binary Image",WINDOW_NORMAL);
       
      //cout<<bw.rows<<endl<<bw.cols<<endl<<"sudip"<<endl;
       Mat dst1, cdst;
       Canny(binary, dst1, 10, 255, 3);
       cvtColor(dst1, cdst, CV_GRAY2BGR);
      contour(binary);
       
        vector<Vec4i> lines;
        HoughLinesP(dst1, lines, 1, CV_PI/180, 50, 50, 10);
        for( size_t i = 0; i < lines.size(); i++ )
        {
          Vec4i l = lines[i];
          line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        }





    //   imshow("source", src);

     // imshow("RGB image", RGB);
     // imshow("HSV Image", hsv);

       //imshow("Binary Image",binary);

       



      //Canny(frame,img,100,240,5);

      waitKey(10);

      }


}



void contour(Mat gray)
 {


     namedWindow( "Validation Gate", 0 );
     namedWindow( "gate_binary", 0 );
	
        double precTick = ticks;
        ticks = (double) cv::getTickCount(); 
 
        double dt = (ticks - precTick) / cv::getTickFrequency(); //Current Ticks count - Precious Ticks Count Divided by Frequency
 
   Mat gate_binary(gray.rows,gray.cols,CV_8UC1,Scalar(0));
     Canny(gray, gray, 10, 40, 3);
/******************CONTOUR DETECTION*********************/
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;
     RNG rng(12345);
     findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
     

     double x1[(src.rows)*(src.cols)], y1[(src.rows)*(src.cols)], vx[(src.rows)*(src.cols)], vy[(src.rows)*(src.cols)];



	//cout<<"18MF10028: "<<contours.size()<<endl;
      for (unsigned int i = 0;  i < contours.size();  i++)
    {
     


     if((contourArea(contours[i])>950))
     {
      
         ////////////////////
	
           vector<Moments> mu(contours.size() );

	//cout<<"sudip_chakraborty: "<<mu[i].m10/mu[i].m00<<" "<<mu[i].m01/mu[i].m00<<endl;
            for( int j = 0; j < contours.size(); j++ )
               { mu[j] = moments( contours[j], false ); }

      /*************** calculation of the center of the gate*******************/
            vector<Point2f> mc( contours.size() );
            for( int j = 0; j < contours.size(); j++ )
               { 
                  mc[j] = Point2f( mu[j].m10/mu[j].m00 , mu[j].m01/mu[j].m00 ); 
               }

             

             if(mc[i].x>200&&mc[i].y<330&&mc[i].x<510)
             {
                Scalar color = Scalar( 255 );
                drawContours( gate_binary, contours, i, color, 2, 8, hierarchy, 0, Point() );

                Scalar color1 = Scalar( 255,0,0 );
                drawContours( src, contours, i, color1, 2, 8, hierarchy, 0, Point() );
		//gate_binary.at<uchar>(mu[i].m10,mu[i].m01)=255;
                cout<<"Center = "<<mc[i].x << ","<< mc[i].y<<endl;
		


                for(int k=mc[i].x-2;k<mc[i].x+1;k++)
                {

                  for(int l=mc[i].y-2;l<mc[i].y+1;l++)
                  {
	
                    src.at<Vec3b>(l+20,k)[1]=255;
                  }
                }



                cout << " Area: " << contourArea(contours[i]) << endl;
             }





      }
    }

      Mat dst1,cdst;
      Canny(gate_binary, dst1, 10, 255, 3);
      cvtColor(dst1, cdst, CV_GRAY2BGR);
      int k109=0;
      
        vector<Vec4i> lines;
        HoughLinesP(dst1, lines, 1, CV_PI/180, 50, 50, 10);
        for( size_t i = 0; i < lines.size(); i++ )
        {
          Vec4i l = lines[i];
          //line( gate_binary, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(144), 3, CV_AA);
          x1[k109]=l[0];
          y1[k109]=l[1];
		vx[k109]=0;
		vy[k109]=0;
          k109++;
          x1[k109]=l[2];
          y1[k109]=l[3];
		vx[k109]=0;
		vy[k109]=0;
          k109++;

        }

        int flag=k109;

      for(int  ij=0;ij<flag;ij++)
      {
        vx[ij]=(x1[ij]-x_pr[ij])/dt;
        x_pr[ij]=x1[ij];
        vy[ij]=(y1[ij]-y_pr[ij])/dt;
        y_pr[ij]=y1[ij];  

      }
/******************************KALMEN FILTERING************************************************************************************/

Matrix4d I, K, A, P, Pp, R;

    /*X is the iteration matrix of state variables x, y, vx, vx, Xp is prediction matrix of X. Similarly Ym is measurement matrix received as input in each iteration and Y
    is the processed matrix Y = C*Ym + Z*/
    Matrix<double, 4, 1> X, Xp, Y, Ym;
    Matrix<double, 4, 2> B;
    Matrix<double, 2, 1> U;
    I <<1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    A <<1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;
    B <<0.5*dt*dt, 0,
        0, 0.5*dt*dt,
        dt, 0,
        0, dt;
    int i = 0;
    char ch = 27;
    
    
    //Initialize covariance matrices
    Pp<<20, 0, 0, 0,
        0, 20, 0, 0,
        0, 0, 10, 0,
        0, 0, 0, 10;
    R<<10, 0, 0, 0,
        0, 10, 0, 0,
        0, 0, 0.5, 0,
        0, 0, 0, 0.5;
    Matrix<double, 4, 1> W;
    W <<10,
        10,
        0,
        0;
    i = 0;
    while (flag--)
        {
          if(i==0)
          {
              X <<x1[i],
                  y1[i],
                  0,
                  0;
              Ym <<x1[i],
                  y1[i],
                  vx[i],
                  vy[i];
          }
          if(i > 0)
          {
              U <<((vx[i] - vx[i-1])/dt),
                  ((vy[i] - vy[i-1])/dt);
          }
          else
          {
              U <<0,
                  0;
          }
          //Predict Xp, Pp
          Xp = A*X  ;
          Pp = A*P*(A.transpose());

          //Calculate Kalman gain
          K = Pp*((Pp + R).inverse());
          Ym <<x1[i],
              y1[i],
              vx[i],
              vy[i];
          Y = Ym;

          //Update X
          X = Xp + K*(Y - Xp);

          //Update Pp
          P = (I - K)*Pp;
          //cout << X << endl << endl;
          //src.at<Vec3b>(round(y[i]) + 500, round(x[i]) + 500)[0] = 255;
          RGB.at<Vec3b>(round(X(1, 0)) , round(X(0, 0)))[1] = 255;
          //img1.at<Vec3b>((round(50*vy[i])) + 700, (round(50*vx[i])+700))[0] = 255;
          //img1.at<Vec3b>(round(50*X(3, 0))+700, round(50*X(2, 0))+700)[1] = 255;
          //cout << X(2, 0) << ' ' << X(3, 0) << endl;
          //cout << x[i] << ' ' << y[i] << ' ' << vx[i] << ' ' << vy[i] << ' ' << endl;
          i++;
          namedWindow("Plot", 0);
          imshow("Plot", src);
          waitKey(10);
        }
    //cout << xd << ' ' << yd << ' ' << vxd << ' ' << vyd << ' ' << endl;
    //cout << xm << ' ' << ym << ' ' << vxm << ' ' << vym << ' ' << endl;
    


/******************************************************************************************************************/        





       


   imshow( "Validation Gate", src );
  imshow( "gate_binary", gate_binary );
  
}
