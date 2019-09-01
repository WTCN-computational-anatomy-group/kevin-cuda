# User manual [DCT\_DST\_OnetoFour(double and single)-
MATLAB\_CUDA\_MEX
# ]

/\*

 \* Discrete Cosine/Sine Transform (DCT/DST and IDCT/IDST one to four-all in one)

 \* DCT/DST and IDCT/IDST I ---\&gt; IV

 \* This CUDA code can handle/work with any type of the input mxArrays,

 \* GPUarray or standard matlab CPU array as input {prhs [0]:= mxGPUArray or CPU Array}

 \* GpuArray/cpuArray output, B=Discrete Transform (A, type of Transform (sine or cosine), type of Transform (direct/inverse), type of DCT/DST or IDCT/IDST, dimensions).

 \* Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London

 \* Welcome Trust Centre for Neuroimaging

 \* Part of the project SPM ([http://www.fil.ion.ucl.ac.uk/spm](http://www.fil.ion.ucl.ac.uk/spm))

 \* Copyright 2018

 \* Kevin Bronik

 \*/

# To compile:

First try the method described here:

[https://uk.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html](https://uk.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html)

After successful compiling running and testing thensimply try following statement (copy and paste in Matlab and enter):

\&gt;\&gt; debug\_Discrete\_Transform(false)

See the file &quot;debug\_Discrete\_Transform.m&quot;

**To compute Discrete Cosine/Sine Transform DCT/DST and inverse Discrete Cosine/Sine Transform IDCT/IDST user can choose/use the following unified syntax:**

**B=Discrete\_Transform** (Input array A, Type of Discrete Transform, Type of Transformation, Type of DCT/DST or IDCT/IDST, Dimensions)

Where

B: = **output array**  **same type as input array**

Input array A: = **array, or gpuArray object**

Type of Discrete Transform: = **sine or cosine**

Type of Transformation: = **direct or inverse**

Type of DCT/DST or IDCT/IDST: = **one, two, three or four**      **(I, II, III, IV)**

Dimensions: = **row or column**           ( **dimension to operate along** )

**Examples:**

**(First example**)

\&gt;\&gt; a = [1, 2, 3; 4, 5, 6; 7, 8, 9];    --- (original input array)

\&gt;\&gt; a=single (a)

a =

  3×3 single matrix

     1     2     3

     4     5     6

     7     8     9

\&gt;\&gt; e=Discrete\_Transform (a, &#39;cosine&#39;, &#39;direct&#39;, &#39;two&#39;, &#39;row&#39;)

                                                      --- (direct transform)

e =

  3×3 single matrix

    6.9282    8.6603   10.3923

   -4.2426   -4.2426   -4.2426

   -0.0000   -0.0000   -0.0000

\&gt;\&gt; d=Discrete\_Transform (e, &#39;cosine&#39;, &#39;inverse&#39;, &#39;two&#39;, &#39;row&#39;)

                                            --- (inverse transform-recovery)

d =

  3×3 single matrix

    1.0000    2.0000    3.0000

    4.0000    5.0000    6.0000

    7.0000    8.0000    9.0000

\&gt;\&gt;

**(Second example**)

\&gt;\&gt; a = single (ones (12, 5,&#39;gpuArray&#39;)); --- (original input array)

\&gt;\&gt; a

a =

  12×5 single gpuArray matrix

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

     1     1     1     1     1

\&gt;\&gt; e=Discrete\_Transform (a, &#39;sine&#39;, &#39;direct&#39;, &#39;three&#39;, &#39;column&#39;)

                                                 --- (direct transform)

e =

  12×5 single gpuArray matrix

    3.1277    1.0668    0.6706   -0.0000    0.0000

    3.1277    1.0668   -0.0000   -0.0000    0.0000

    3.1277    1.0668   -0.0000   -0.0000    0.4118

    3.1277    0.0000   -0.0000   -0.0000    0.4118

    3.1277    0.0000   -0.0000    0.4419    0.4118

         0    0.0000   -0.0000    0.4419    0.4118

         0    0.0000    0.5146    0.4419    0.4118

         0    0.0000    0.5146    0.4419         0

         0    0.6706    0.5146    0.4419         0

         0    0.6706    0.5146    0.0000         0

    1.0668    0.6706    0.5146    0.0000         0

    1.0668    0.6706   -0.0000    0.0000         0

\&gt;\&gt; f=Discrete\_Transform (e, &#39;sine&#39;, &#39;inverse&#39;, &#39;three&#39;, &#39;column&#39;)

                                            --- (inverse transform-recovery)

f =

  12×5 single gpuArray matrix

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

    1.0000    1.0000    1.0000    1.0000    1.0000

