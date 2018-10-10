
 function debug_DST_II_Column_Inverse(bDebug)
 
% mexcuda -largeArrayDims DST_II_Column_Inverse.cu

    if(bDebug)
      mexcuda -c DST_II_Column_Inverse.cu 
    else
      mexcuda -v DST_II_Column_Inverse.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_II_Column_Inverse(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_II_Column_Inverse(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 