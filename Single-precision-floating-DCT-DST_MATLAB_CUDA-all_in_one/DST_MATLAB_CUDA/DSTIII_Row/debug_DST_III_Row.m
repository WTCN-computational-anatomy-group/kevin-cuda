
 function debug_DST_III_Row(bDebug)
 
% mexcuda -largeArrayDims DST_III_Row.cu

    if(bDebug)
      mexcuda -c DST_III_Row.cu 
    else
      mexcuda -v DST_III_Row.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_III_Row(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_III_Row(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 