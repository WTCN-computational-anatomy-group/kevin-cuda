%
 % Three dimensional Discrete Cosine/Sine Transform(DCT/DST and IDCT/IDST one to four-all in one)
 % DCT/DST and IDCT/IDST I ---> IV
 % This MATLAb-CUDA  Wrapper function can handle/work with  any type of the input mxArrays, 
 % GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 % gpuArray output, B=Discrete_Transform_3D(A, , type of Transform (sine or cosine), type of Transform(direct/inverse), type of DCT/DST or IDCT/IDST, dimensions).
 % Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 % Wellcome Trust Centre for Neuroimaging
 % Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 % Copyright 2018
 % Kevin Bronik
 %
 
 function dct3Coefs = Discrete_Transform_3D(data,typ1,typ2,type3,DIM)


switch DIM
    
    case 'full'
        [ex,wy,zy]=size(data);
        if ex==1|| wy==1||zy==1
        error('Please try lower dimension for Discrete Transform');
        end
    case 'row'
        data=permute(data, [3 2 1]);
        [ex,wy,zy]=size(data);
         if ex==1|| wy==1||zy==1
        error('Please try lower dimension for Discrete Transform');
        end
    case 'column'
        data=permute(data, [1 3 2]);
        [ex,wy,zy]=size(data);
        if ex==1|| wy==1||zy==1
        error('Please try lower dimension for Discrete Transform');
        end
    case 'third'
        data=permute(data, [1 2 3]);
        [ex,wy,zy]=size(data);
        if ex==1|| wy==1||zy==1
        error('Please try lower dimension for Discrete Transform');
        end
    otherwise 
     error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);       

end


%....................................................................CPUarray
if ~isa(data,'gpuArray' )
    
% direct...............................................................    
  switch  typ2  
      case 'direct'
dct3Coefs=single(zeros(ex,wy,zy));
for i=1:zy
    tempData=data(:,:,i);
   % dct2Coefs=dct2(tempData);
   switch typ1
     case  'cosine'
       
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                    case 'row'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'column'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'third'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end    
                   
                 otherwise
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full,, not  %s.',DIM);       
                    
             end % end DIM
       
   case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                  case 'row'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'column'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'third'
                     switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end      
                    
                    
                    
                otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
                    end  % end DIM
       
       
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
   end

   % dct3Coefs(:,:,i)=dct2Coefs;
     dct3Coefs(:,:,i)=dct2Coefs;
end

switch DIM
    
    case 'full'
for i=1:ex
    for j=1:wy
        tempDCT=[];
        for k=1:zy
            tempDCT=[tempDCT,dct3Coefs(i,j,k)];
        end
        %dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
      switch typ1
     case  'cosine'
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'three' , 'row');
                        case 'four'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
          case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'three' , 'row');
                        case 'four'
    
   dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                   otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
       
      otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
     end
        
        for k=1:zy
            dct3Coefs(i,j,k)=dctOverdct(1,k);
        end
    end
end

    case 'row'
        %result=ipermute(coefficients_of_3d_idct, [3 2 1]);
        dct3Coefs=ipermute(dct3Coefs, [3 2 1]);
        
    case 'column'
        dct3Coefs=permute(dct3Coefs, [1 3 2]);
        
    case 'third'
        dct3Coefs=permute(dct3Coefs, [1 2 3]);

end  % end DIM
% END direct...............................................................
% inverse..............................................................
      case 'inverse'
dct3Coefs=single(zeros(ex,wy,zy));
for i=1:zy
    tempData=data(:,:,i);
   % dct2Coefs=dct2(tempData);
   switch typ1
     case  'cosine'
       
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                    case 'row'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'column'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'third'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end    
                   
                 otherwise
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full,, not  %s.',DIM);       
                    
             end % end DIM
       
   case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                  case 'row'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'column'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'third'
                     switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end      
                    
                    
                    
                otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
                    end  % end DIM
       
       
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
   end

   % dct3Coefs(:,:,i)=dct2Coefs;
     dct3Coefs(:,:,i)=dct2Coefs;
end

switch DIM
    
    case 'full'
for i=1:ex
    for j=1:wy
        tempDCT=[];
        for k=1:zy
            tempDCT=[tempDCT,dct3Coefs(i,j,k)];
        end
        %dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
      switch typ1
     case  'cosine'
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'three' , 'row');
                        case 'four'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
          case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'three' , 'row');
                        case 'four'
    
   dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                   otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
       
      otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
     end
        
        for k=1:zy
            dct3Coefs(i,j,k)=dctOverdct(1,k);
        end
    end
end

    case 'row'
        %result=ipermute(coefficients_of_3d_idct, [3 2 1]);
        dct3Coefs=ipermute(dct3Coefs, [3 2 1]);
        
    case 'column'
        dct3Coefs=permute(dct3Coefs, [1 3 2]);
        
    case 'third'
        dct3Coefs=permute(dct3Coefs, [1 2 3]);

end

      otherwise
       error('Input(THIRD ARGUMENT) must be direct or inverse, not  %s.',typ2);
% END inverse..............................................................
end % typ2
%....................................................................END CPUarray
%....................................................................GPUarray
elseif  isa(data,'gpuArray' )
    
    switch  typ2  
      case 'direct'
dct3Coefs=single(zeros(ex,wy,zy, 'gpuArray'));
for i=1:zy
    tempData=gpuArray(data(:,:,i));
   % dct2Coefs=dct2(tempData);
   switch typ1
     case  'cosine'
       
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                    case 'row'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'column'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'third'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end    
                   
                 otherwise
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full,, not  %s.',DIM);       
                    
             end % end DIM
       
   case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                  case 'row'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'column'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'third'
                     switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end      
                    
                    
                    
                otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
                    end  % end DIM
       
       
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
   end

   % dct3Coefs(:,:,i)=dct2Coefs;
     dct3Coefs(:,:,i)=dct2Coefs;
end

switch DIM
    
    case 'full'
for i=1:ex
    for j=1:wy
        tempDCT=[];
        for k=1:zy
            tempDCT=[tempDCT,dct3Coefs(i,j,k)];
        end
        %dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
      switch typ1
     case  'cosine'
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'three' , 'row');
                        case 'four'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
          case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'three' , 'row');
                        case 'four'
    
   dctOverdct=Discrete_Transform(tempDCT, 'sine', 'direct', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                   otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
       
      otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
     end
        
        for k=1:zy
            dct3Coefs(i,j,k)=dctOverdct(1,k);
        end
    end
end

    case 'row'
        %result=ipermute(coefficients_of_3d_idct, [3 2 1]);
        dct3Coefs=ipermute(dct3Coefs, [3 2 1]);
        
    case 'column'
        dct3Coefs=permute(dct3Coefs, [1 3 2]);
        
    case 'third'
        dct3Coefs=permute(dct3Coefs, [1 2 3]);

end  % end DIM
% END direct...............................................................
% inverse..............................................................
      case 'inverse'
dct3Coefs=single(zeros(ex,wy,zy, 'gpuArray'));
for i=1:zy
    tempData=gpuArray(data(:,:,i));
   % dct2Coefs=dct2(tempData);
   switch typ1
     case  'cosine'
       
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                    case 'row'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'column'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end
                   case 'third'
                        switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                        end    
                   
                 otherwise
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full,, not  %s.',DIM);       
                    
             end % end DIM
       
   case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                  case 'row'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'column'
                    switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 case 'third'
                     switch  type3
                        case 'one'
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    dct2Coefs=Discrete_Transform(Discrete_Transform(tempData, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end      
                    
                    
                    
                otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
                    end  % end DIM
       
       
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
   end

   % dct3Coefs(:,:,i)=dct2Coefs;
     dct3Coefs(:,:,i)=dct2Coefs;
end

switch DIM
    
    case 'full'
for i=1:ex
    for j=1:wy
        tempDCT=[];
        for k=1:zy
            tempDCT=[tempDCT,dct3Coefs(i,j,k)];
        end
        %dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'direct', 'one' , 'row');
      switch typ1
     case  'cosine'
             switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'three' , 'row');
                        case 'four'
    
    dctOverdct=Discrete_Transform(tempDCT, 'cosine', 'inverse', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                 otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
          case 'sine' 
                    switch  DIM
             case 'full'  
                    switch  type3
                        case 'one'
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'one' , 'row');
                        case 'two'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'two' , 'row');
                        case 'three'
    
    dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'three' , 'row');
                        case 'four'
    
   dctOverdct=Discrete_Transform(tempDCT, 'sine', 'inverse', 'four' , 'row');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
                   otherwise 
                         error(' Input(FIFTH ARGUMENT) must be row or column or third or full, not  %s.',DIM);
             end
       
       
      otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',typ1);
     end
        
        for k=1:zy
            dct3Coefs(i,j,k)=dctOverdct(1,k);
        end
    end
end

    case 'row'
        %result=ipermute(coefficients_of_3d_idct, [3 2 1]);
        dct3Coefs=ipermute(dct3Coefs, [3 2 1]);
        
    case 'column'
        dct3Coefs=permute(dct3Coefs, [1 3 2]);
        
    case 'third'
        dct3Coefs=permute(dct3Coefs, [1 2 3]);

end

      otherwise
       error('Input(THIRD ARGUMENT) must be direct or inverse, not  %s.',typ2);
% END inverse..............................................................
end  
    
    
%....................................................................END GPUarray 


else
    error('Input(FIRST ARGUMENT) must be array, or gpuArray object, not  %s.',data);    
end  % end  of if ~isa(data,'gpuArray' )


 end %function


 