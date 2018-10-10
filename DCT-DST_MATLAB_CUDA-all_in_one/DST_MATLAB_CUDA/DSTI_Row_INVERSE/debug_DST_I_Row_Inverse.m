
 function debug_DST_I_Row_Inverse(bDebug)
 
% mexcuda -largeArrayDims DST_I_Row_Inverse.cu

    if(bDebug)
      mexcuda -c DST_I_Row_Inverse.cu 
    else
      mexcuda -v DST_I_Row_Inverse.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_I_Row_Inverse(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_I_Row_Inverse(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 