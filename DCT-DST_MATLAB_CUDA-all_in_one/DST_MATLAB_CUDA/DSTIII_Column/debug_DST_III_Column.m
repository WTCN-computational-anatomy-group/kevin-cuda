
 function debug_DST_III_Column(bDebug)
 
% mexcuda -largeArrayDims DST_III_Column.cu

    if(bDebug)
      mexcuda -c DST_III_Column.cu 
    else
      mexcuda -v DST_III_Column.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_III_Column(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_III_Column(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 