
 function debug_DCT_III_Column(bDebug)
 
% mexcuda -largeArrayDims DCT_III_Column.cu

    if(bDebug)
      mexcuda -c DCT_III_Column.cu 
    else
      mexcuda -v DCT_III_Column.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DCT_III_Column(x)
    
%  or test this
%     x = ones(4,4);
%     y = DCT_III_Column(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 