%
 % Two dimensional (full) Discrete Cosine/Sine Transform(DCT/DST and IDCT/IDST one to four-all in one)
 % DCT/DST and IDCT/IDST I ---> IV
 % This MATLAb-CUDA  Wrapper function can handle/work with  any type of the input mxArrays, 
 % GPUarray or standard matlab CPU array as input {prhs[0] := mxGPUArray or CPU Array}
 % gpuArray output, B=Discrete_Transform_2D(A, , type of Transform (sine or cosine), type of Transform(direct/inverse), type of DCT/DST or IDCT/IDST).
 % Developed at UCL, Institute of Neurology, 12 Queen Square, WC1N 3AR, London
 % Wellcome Trust Centre for Neuroimaging
 % Part of the project SPM(http://www.fil.ion.ucl.ac.uk/spm)
 % Copyright 2018
 % Kevin Bronik
 %
 
 function DctDst2Out = Discrete_Transform_2D(data,type1,type2,type3)

 [ex,wy]=size(data);
        if ex==1|| wy==1
        error('Please try lower dimension for Discrete Transform');
        end

 

%....................................................................CPUarray
if ~isa(data,'gpuArray' )
  DctDst2Out =single(zeros(ex,wy)); 
% direct...............................................................    
switch   type2  
      
 case 'direct'


   switch type1
       
     case  'cosine'
       
  
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end

       
   case 'sine' 
 
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
   
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',type1);
   end
% direct ......................

case 'inverse'
   
      switch type1
       
             case  'cosine'
       
  
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end

       
   case 'sine' 
 
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
   
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',type1);
   end
          
      otherwise
       error('Input(THIRD ARGUMENT) must be direct or inverse, not  %s.',type2);

end
% GPU..............................................

elseif  isa(data,'gpuArray' )
    
  DctDst2Out =single(zeros(ex,wy,'gpuArray'));  
% direct...............................................................    
switch   type2  
      
 case 'direct'


   switch type1
       
     case  'cosine'
       
  
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'one' , 'row'), 'cosine', 'direct', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'two' , 'row'), 'cosine', 'direct', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'three' , 'row'), 'cosine', 'direct', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'direct', 'four' , 'row'), 'cosine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end

       
   case 'sine' 
 
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'one' , 'row'), 'sine', 'direct', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'two' , 'row'), 'sine', 'direct', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'three' , 'row'), 'sine', 'direct', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'direct', 'four' , 'row'), 'sine', 'direct', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
   
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',type1);
   end
% direct ......................

case 'inverse'
   
      switch type1
       
             case  'cosine'
       
  
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'one' , 'row'), 'cosine', 'inverse', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'two' , 'row'), 'cosine', 'inverse', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'three' , 'row'), 'cosine', 'inverse', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'cosine', 'inverse', 'four' , 'row'), 'cosine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end

       
   case 'sine' 
 
                    switch  type3
                        case 'one'
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'one' , 'row'), 'sine', 'inverse', 'one' , 'column');
                        case 'two'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'two' , 'row'), 'sine', 'inverse', 'two' , 'column');
                        case 'three'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'three' , 'row'), 'sine', 'inverse', 'three' , 'column');
                        case 'four'
    
    DctDst2Out=Discrete_Transform(Discrete_Transform(data, 'sine', 'inverse', 'four' , 'row'), 'sine', 'inverse', 'four' , 'column');
                      otherwise 
                         error(' Input(FOURTH ARGUMENT) must be one or two or three or four, not  %s.',type3);
                    end
   
   otherwise
       error('Input(SECOND ARGUMENT) must be cosine or sine, not  %s.',type1);
   end
          
      otherwise
       error('Input(THIRD ARGUMENT) must be direct or inverse, not  %s.',type2);

end   
    
    
else
    error('Input(FIRST ARGUMENT) must be array, or gpuArray object, not  %s.',data);     
    
end   



 end %function