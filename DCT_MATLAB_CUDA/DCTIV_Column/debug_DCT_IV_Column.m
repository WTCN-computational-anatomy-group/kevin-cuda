
 function debug_DCT_IV_Column(bDebug)
 
% mexcuda -largeArrayDims DCT_IV_Column.cu

    if(bDebug)
      mexcuda -c DCT_IV_Column.cu 
    else
      mexcuda -v DCT_IV_Column.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DCT_IV_Column(x)
    
%  or test this
%     x = ones(4,4);
%     y = DCT_IV_Column(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 