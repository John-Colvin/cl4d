rdmd --build-only -version=CL_VERSION_1_1 -wi -gc -L-L/opt/AMDAPP/lib/x86_64 -L-lOpenCL -ofvectorAdd vectorAdd.d
#rdmd --force --build-only --chatty -version=CL_VERSION_1_1 -release -O -inline -d -w -I../Derelict2/DerelictGL/ -I../Derelict2/DerelictSDL/ -I../Derelict2/DerelictUtil/ -L-L/opt/AMDAPP/lib/x86_64 -L-lOpenCL -ofCLGLInterop CLGLInterop.d
