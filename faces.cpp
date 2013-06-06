#include "Poco/URIStreamOpener.h"
#include "Poco/StreamCopier.h"
#include "Poco/Path.h"
#include "Poco/URI.h"
#include "Poco/Exception.h"
#include "Poco/Net/HTTPStreamFactory.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <memory>

using namespace std;
using namespace cv;
using Poco::URIStreamOpener;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;
using Poco::Net::HTTPStreamFactory;

char buf[4096];

extern "C" char* detect_faces(char* cascade_file, char* input_file);

int main(int argc, char** argv) {
  if(argc<2){
    fprintf(stderr, "usage:\n%s <image>\n%s <image>\n", argv[0], argv[0]);
    exit(-1);
  }
  printf("%s", detect_faces(argv[1], argv[2]));
  exit(0);
}

static bool factoryLoaded = false;

class webImageLoader {
  public :
  Mat loadFromURL(string url){

    //Don't register the factory more than once
    if(!factoryLoaded){
      HTTPStreamFactory::registerFactory();
      factoryLoaded = true;
    }

    //Specify URL and open input stream
    URI uri(url);
    auto_ptr<istream> pStr(URIStreamOpener::defaultOpener().open(uri));

    //Copy image to our string and convert to Mat
    string str;
    StreamCopier::copyToString(*pStr.get(), str);
    vector<char> data( str.begin(), str.end() );
    Mat data_mat(data);
    Mat image(imdecode(data_mat,1));

    return image;
  }
};

Mat stringtoMat(string file) {
  Mat image;

  if (file.compare(file.size()-4,4,".gif")==0) {

    return image;

  } else if (file.compare(0,7,"http://")==0 || file.compare(0,8,"https://")==0) {

    webImageLoader loader;
    try {
      image = loader.loadFromURL(file);
    } catch (exception& e) {
    }

    return image;

  } else {

    image = imread(file,1); // Try if the image path is in the local machine
    return image;

  }
}


char* detect_faces(char* cascade_file, char* input_file) {
  *buf = 0;
  Mat imgbw;
  Mat image = stringtoMat(input_file);

  if(image.empty()) return buf;

  CascadeClassifier cascade;
  if(!cascade.load(cascade_file)) return buf; //load classifier cascade

  cvtColor(image, imgbw, CV_BGR2GRAY); //create a grayscale copy
  equalizeHist(imgbw, imgbw); //apply histogram equalization
  vector<Rect> faces;
  cascade.detectMultiScale(imgbw, faces, 1.2, 2); //detect faces

  for(unsigned int i = 0; i < faces.size(); i++){
    Rect f = faces[i];
    //draw rectangles on the image where faces were detected
    rectangle(image, Point(f.x, f.y), Point(f.x + f.width, f.y + f.height), Scalar(255, 0, 0), 4, 8);
    //fill buffer with easy to parse face representation
    sprintf(buf + strlen(buf), "%i;%i;%i;%i\n", f.x, f.y, f.width, f.height);
  }

  return buf;
}