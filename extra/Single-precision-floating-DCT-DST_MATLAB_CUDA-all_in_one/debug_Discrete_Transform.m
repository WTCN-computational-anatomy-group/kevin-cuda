
 function debug_Discrete_Transform(bDebug)
 
% mexcuda -largeArrayDims  Discrete_Transform.cu  DCT_I_Column.cu DCT_I_Column_Inverse.cu  DCT_I_Row.cu  DCT_I_Row_Inverse.cu DCT_II_Column.cu DCT_II_Column_Inverse.cu  DCT_II_Row.cu  DCT_II_Row_Inverse.cu  DCT_III_Column.cu DCT_III_Column_Inverse.cu  DCT_III_Row.cu  DCT_III_Row_Inverse.cu  DCT_IV_Column.cu DCT_IV_Column_Inverse.cu  DCT_IV_Row.cu  DCT_IV_Row_Inverse.cu
% DST_I_Column.cu DST_I_Column_Inverse.cu  DST_I_Row.cu  DST_I_Row_Inverse.cu DST_II_Column.cu DST_II_Column_Inverse.cu  DST_II_Row.cu  DST_II_Row_Inverse.cu  DST_III_Column.cu DST_III_Column_Inverse.cu  DST_III_Row.cu  DST_III_Row_Inverse.cu  DST_IV_Column.cu DST_IV_Column_Inverse.cu  DST_IV_Row.cu  DST_IV_Row_Inverse.cu
  %mexcuda -largeArrayDims  Discrete_Transform.cu  DCT_I_Column.cu DCT_I_Column_Inverse.cu  DCT_I_Row.cu  DCT_I_Row_Inverse.cu DCT_II_Column.cu DCT_II_Column_Inverse.cu  DCT_II_Row.cu  DCT_II_Row_Inverse.cu  DCT_III_Column.cu DCT_III_Column_Inverse.cu  DCT_III_Row.cu  DCT_III_Row_Inverse.cu  DCT_IV_Column.cu DCT_IV_Column_Inverse.cu  DCT_IV_Row.cu  DCT_IV_Row_Inverse.cu  DST_I_Column.cu DST_I_Column_Inverse.cu  DST_I_Row.cu  DST_I_Row_Inverse.cu DST_II_Column.cu DST_II_Column_Inverse.cu  DST_II_Row.cu  DST_II_Row_Inverse.cu  DST_III_Column.cu DST_III_Column_Inverse.cu  DST_III_Row.cu  DST_III_Row_Inverse.cu  DST_IV_Column.cu DST_IV_Column_Inverse.cu  DST_IV_Row.cu  DST_IV_Row_Inverse.cu 
   if(bDebug)
      mexcuda -c Discrete_Transform.cu  DCT_I_Column.cu DCT_I_Column_Inverse.cu  DCT_I_Row.cu  DCT_I_Row_Inverse.cu DCT_II_Column.cu DCT_II_Column_Inverse.cu  DCT_II_Row.cu  DCT_II_Row_Inverse.cu  DCT_III_Column.cu DCT_III_Column_Inverse.cu  DCT_III_Row.cu  DCT_III_Row_Inverse.cu  DCT_IV_Column.cu DCT_IV_Column_Inverse.cu  DCT_IV_Row.cu  DCT_IV_Row_Inverse.cu  DST_I_Column.cu DST_I_Column_Inverse.cu  DST_I_Row.cu  DST_I_Row_Inverse.cu DST_II_Column.cu DST_II_Column_Inverse.cu  DST_II_Row.cu  DST_II_Row_Inverse.cu  DST_III_Column.cu DST_III_Column_Inverse.cu  DST_III_Row.cu  DST_III_Row_Inverse.cu  DST_IV_Column.cu DST_IV_Column_Inverse.cu  DST_IV_Row.cu  DST_IV_Row_Inverse.cu  
    else
      mexcuda -v Discrete_Transform.cu  DCT_I_Column.cu DCT_I_Column_Inverse.cu  DCT_I_Row.cu  DCT_I_Row_Inverse.cu DCT_II_Column.cu DCT_II_Column_Inverse.cu  DCT_II_Row.cu  DCT_II_Row_Inverse.cu  DCT_III_Column.cu DCT_III_Column_Inverse.cu  DCT_III_Row.cu  DCT_III_Row_Inverse.cu  DCT_IV_Column.cu DCT_IV_Column_Inverse.cu  DCT_IV_Row.cu  DCT_IV_Row_Inverse.cu  DST_I_Column.cu DST_I_Column_Inverse.cu  DST_I_Row.cu  DST_I_Row_Inverse.cu DST_II_Column.cu DST_II_Column_Inverse.cu  DST_II_Row.cu  DST_II_Row_Inverse.cu  DST_III_Column.cu DST_III_Column_Inverse.cu  DST_III_Row.cu  DST_III_Row_Inverse.cu  DST_IV_Column.cu DST_IV_Column_Inverse.cu  DST_IV_Row.cu  DST_IV_Row_Inverse.cu 
    end
 
%     test this 
    x = ones(4,4,'gpuArray');
    d=Discrete_Transform(x, 'cosine', 'inverse', 'one' , 'row')
    
%  or test this
%     x = ones(4,4);
%     y = Discrete_Transform(x, 'cosine', 'inverse', 'one' , 'row')
 
 
 
    disp('finished without error. Hit any key to exit');
    clear mex;pause;exit;
 