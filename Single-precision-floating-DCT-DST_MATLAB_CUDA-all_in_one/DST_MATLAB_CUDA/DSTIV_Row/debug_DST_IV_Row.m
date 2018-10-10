
 function debug_DST_IV_Row(bDebug)
 
% mexcuda -largeArrayDims DST_IV_Row.cu

    if(bDebug)
      mexcuda -c DST_IV_Row.cu 
    else
      mexcuda -v DST_IV_Row.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_IV_Row(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_IV_Row(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 