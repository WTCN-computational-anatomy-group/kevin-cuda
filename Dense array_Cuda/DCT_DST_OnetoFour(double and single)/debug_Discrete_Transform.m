
 function debug_Discrete_Transform(bDebug)
 
% /*
%  * Discrete Cosine/Sine Transform(DCT/DST and IDCT/IDST one to four-all in one)
%  * DCT/DST and IDCT/IDST I ---> IV
%  * This CUDA code can handle/work with  any type of the input mxArrays, 
%  * GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
%  * GpuArray/cpuArray output, B=Discrete_Transform(A, , type of Transform (sine or cosine), type of Transform(direct/inverse), type of DCT/DST or IDCT/IDST, dimensions).
%  * Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
%  * Wellcome Trust Centre for Neuroimaging
%  * Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
%  * Copyright 2018
%  * Kevin Bronik
%  */

if ismac
    % Code to run on Mac plaform
elseif isunix
    % checks

if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-9.0 /usr/local/cuda"')
end
end

newpath = fileparts(mfilename('fullpath'));
cd(newpath);
delete Discrete_Transform.mex*
    if(bDebug)
 
mmc  -g '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include'  Discrete_Transform.cu
    else

mmc  '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64'   '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include'  Discrete_Transform.cu
    end
 
%     test this 
    disp('Start testing...');
    OriginalMatrix = ones(4,4,'gpuArray');
    OriginalMatrix=single(OriginalMatrix)
    disp('Calculating Discrete Cosine Transform in row wise (DCT two)...');
    TransformedMatrix=Discrete_Transform(OriginalMatrix, 'cosine', 'direct', 'two' , 'row')
    
    disp('Recovering the original matrix...');
    disp('Calculating Inverse Discrete Cosine Transform in row wise (inverse DCT two)...');
    RecoveredMatrix=Discrete_Transform(TransformedMatrix, 'cosine', 'inverse', 'two' , 'row')
    
%  or test this
%     x = ones(4,4);
%     y = Discrete_Transform(x, 'cosine', 'inverse', 'one' , 'row')
 
 
 
    disp('finished without error.');
    clear mexcuda;
 