Compile options
------------
```
g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/lib -L/usr/local/lib -fpic -Wall -c "faces.cpp" -lPocoNet -lPocoFoundation -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_objdetect;
g++ -shared -I/usr/local/include/opencv -I/usr/local/include/opencv2 -o libfaces.so faces.o -L/usr/local/lib -lPocoNet -lPocoFoundation -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_objdetect;
g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -o faces faces.o -L/usr/local/lib -lPocoNet -lPocoFoundation -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_objdetect
```