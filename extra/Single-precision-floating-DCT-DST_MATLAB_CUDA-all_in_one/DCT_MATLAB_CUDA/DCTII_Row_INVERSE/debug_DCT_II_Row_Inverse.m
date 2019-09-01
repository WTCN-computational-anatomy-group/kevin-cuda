
 function debug_DCT_II_Row_Inverse(bDebug)
 
% mexcuda -largeArrayDims DCT_II_Row_Inverse.cu

    if(bDebug)
      mexcuda -c DCT_II_Row_Inverse.cu 
    else
      mexcuda -v DCT_II_Row_Inverse.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    y = DCT_II_Row_Inverse(x)
    
%  or test this
%     x = ones(4,4);
%     y = DCT_II_Row_Inverse(x)
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 