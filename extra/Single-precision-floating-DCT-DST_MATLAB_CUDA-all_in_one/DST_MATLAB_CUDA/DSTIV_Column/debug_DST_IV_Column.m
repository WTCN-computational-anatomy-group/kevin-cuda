
 function debug_DST_IV_Column(bDebug)
 
% mexcuda -largeArrayDims DST_IV_Column.cu

    if(bDebug)
      mexcuda -c DST_IV_Column.cu 
    else
      mexcuda -v DST_IV_Column.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_IV_Column(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_IV_Column(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 