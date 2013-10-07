cl4d
====

object-oriented wrapper for the OpenCL C API written in the D programming language

This is a fork containing work from Vauru's fork with several modifications. The original code is by Trass3r.

The recommended usage is to list as a depedency in dub.


To build manually, use dub. E.g.
````sh
dub build --build=release
````

To run an example, go to the relevant subdirectorty and run dub. You must have built the core library already.

If you need to build without dub, take a look at the package.json for dependencies and then simply compile the
entire contents of source/cl4d as a library using the toolchain of your choice.
