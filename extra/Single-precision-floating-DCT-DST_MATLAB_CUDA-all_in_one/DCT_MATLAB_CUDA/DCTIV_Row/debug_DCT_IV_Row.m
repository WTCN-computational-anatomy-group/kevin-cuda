
 function debug_DCT_IV_Row(bDebug)
 
% mexcuda -largeArrayDims DCT_IV_Row.cu

    if(bDebug)
      mexcuda -c DCT_IV_Row.cu 
    else
      mexcuda -v DCT_IV_Row.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DCT_IV_Row(x)
    
%  or test this
%     x = ones(4,4);
%     y = DCT_IV_Row(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 