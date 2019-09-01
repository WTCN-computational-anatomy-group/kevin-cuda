
 function debug_DST_I_Column(bDebug)
 
% mexcuda -largeArrayDims DST_I_Column.cu

    if(bDebug)
      mexcuda -c DST_I_Column.cu 
    else
      mexcuda -v DST_I_Column.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DST_I_Column(x)
    
%  or test this
%     x = ones(4,4);
%     y = DST_I_Column(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 