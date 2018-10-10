function varargout = spm_shoot_greens(varargin)
% Build and apply FFT of Green's function (to map from momentum to velocity)
% FORMAT v = spm_shoot_greens(m,K,prm)
% m    - Momentum field n1*n2*n3*3 (single prec. float)
% K    - Fourier transform representation of Green's function
%        - either size n1*n2*n3 or n1*n2*n3*3*3
% prm  - Differential operator parameters (3 voxel sizes, 5 hyper-parameters)
%        - only needed when K is of size n1*n2*n3, in which case, voxel sizes
%          are necessary for dealing with each component individually
% v    - velocity field
%
% FORMAT [K,ld] = spm_shoot_greens('kernel',dm,prm)
% dm  - dimensions n1*n2*n3
% prm - Differential operator parameters (3 voxel sizes, 5 hyper-parameters)
% K   - Fourier transform representation of Green's function
%        - either size n1*n2*n3 or n1*n2*n3*3*3
% ld(1)  - Log determinant of operator
% ld(2)  - Number of degrees of freedom
%
%________________________________________________________
% (c) Wellcome Trust Centre for NeuroImaging (2012)

% John Ashburner
% $Id: spm_shoot_greens.m 7136 2017-07-18 10:51:48Z john $

%spm_diffeo('boundary',0);
    if nargin < 4
        bnd = 0;
    else
        bnd = varargin{4};
    end

    spm_diffeo('boundary',bnd);
    prm = varargin{3};
    Delta1=prm(1);
    Delta2=prm(2);
    Delta3=prm(3);
    Dispabs=prm(4);
    MembLambda=prm(5);
    BendLambda=prm(6);
    ElasticMu=prm(7);
    ElasticLambda=prm(8);

if nargin==4 && isa(varargin{1},'char') && strcmp(varargin{1},'kernel'),
    d   = varargin{2};

    
    F = spm_diffeo('kernel',d,prm);
    if size(F,4) == 1,
        % The differential operator is symmetric, so the Fourier transform should be real
        %XT=real(fftn(F))
    %param s[0:3] Voxel sizes
       [N,M,K]=size(F);
      switch bnd
        case 0   % Circulant
            F  = 1./real(fftn(F));
        case 1   % Neumann
         if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
           

        %Parameter of the absolute displacement penalty plus Gamma
        Gamma=2*MembLambda*(Delta2^(-2))+Dispabs; 
        if isvector(F) % 1D
            
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));  
        F  = 1./(wc);
        
        elseif ismatrix(F) %2D
        
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = ((1:M)'-1)/M*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as));
        F  = 1./(wc'+ws);
        
        else %3D
        Gamma=Gamma+2*MembLambda*(Delta1^(-2));  
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = ((1:M)'-1)/M*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as)); 
        
            
        acz  = ((1:K)'-1)/K*pi;
        wcz  = 2*MembLambda*(Delta1^(-2))*(2-cos(acz));    
            
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
        Alpha= BendLambda*(Delta2^(-4));    
        Gamma= 6*BendLambda*(Delta1^(-4))+Dispabs;

        if isvector(F) % 1D
        
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2); 
        F  = 1./(wc);
        
        elseif ismatrix(F) %2D 
            
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = ((1:M)'-1)/M*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);   
        
        F  = 1./(wc'+ws);    
             
        else %3D
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
         Gamma= Gamma+6*BendLambda*(Delta3^(-4))+Dispabs;
         Beta=  4*BendLambda*(Delta1^(-2))*(Delta2^(-2))+4*BendLambda*(Delta3^(-2))*(Delta2^(-2));
         Xe=4*BendLambda*(Delta2^(-4))+4*BendLambda*(Delta2^(-2)*Delta3^(-2))-4*BendLambda*(Delta1^(-4))-4*BendLambda*(Delta1^(-2)*Delta3^(-2));
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = ((1:M)'-1)/M*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);
         
        acz  = ((1:K)'-1)/K*pi;
        acxz=2*(2*((cos(acz)).^2))-1;
        acyz=2*cos(2*acz);
        acxyz=2*cos(acz);
        acz  = 2*(1-cos(acz));
        Teta=  2*BendLambda*(Delta3^(-2))*(Delta2^(-2));
        Eta= (-1)*BendLambda*(Delta2^(-4))+BendLambda*(Delta3^(-4));
        wcz  = Xe*acxyz+Eta*acyz+Teta*acxz+Beta*acz+Alpha*acz.^2;  
        
        
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end         
            
              
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
        Alpha1= (2*ElasticMu+ElasticLambda)* (Delta1^(-2));  
        Beta1=   ElasticMu*(Delta2^(-2));  
        Gamma1=  (-1)*(2*ElasticMu+ElasticLambda)* (Delta1^(-2));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta1^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta1^(-2));  
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2));
        
        
        if isvector(F) % 1D
            
        if size(F,2)==1
        NN=size(F,1);    
        ac  =((1:NN)'-1)/NN*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma1*acx+Beta1*ac+Alpha1*ac; 
        F  = 1./(wc);    
        end
        
        if size(F,1)==1 
        NN=size(F,2);    
        ac  = ((1:NN)'-1)/NN*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac; 
        F  = 1./(wc);    
        end
        
        elseif ismatrix(F) %2D 
       
        ac  = ((1:N)'-1)/N*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =((1:M)'-1)/M*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy; 
            
        F  = 1./(wc'+ws);   
              
        else %3D
            
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1);
         
        ac  = ((1:N)'-1)/N*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta3^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2))+ElasticMu*(Delta3^(-2));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =((1:M)'-1)/M*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        Beta1=   ElasticMu*(Delta2^(-2))+ElasticMu*(Delta3^(-2));
        Gamma1=  Gamma1+ElasticMu*(Delta3^(-2));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy;    
            
        acz  = ((1:K)'-1)/K*pi;
        acx=2*cos(acz);
        acy=2*(2*((cos(acz)).^2))-1;
        acz  = 2*(1-cos(acz));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta3^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta3^(-2));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta1^(-2));
        Gamma2=  ElasticMu*(Delta3^(-2))+ElasticMu*(Delta1^(-2));
        wcz  = Gamma2*acx+Beta2*acz+Alpha3*acz+Alpha2*acy;    
            
        lambda = kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
                 
        end    
        
    
        else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
       end   
        case 2   % Dirichlet
            
         if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
           

        %Parameter of the absolute displacement penalty plus Gamma
        Gamma=2*MembLambda*(Delta2^(-2))+Dispabs; 
        if isvector(F) % 1D
            
        ac  = (1:N)'/(N+1)*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));  
        F  = 1./(wc);
        
        
        elseif ismatrix(F) %2D
        
        ac  = (1:N)'/(N+1)*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = (1:M)'/(M+1)*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as));
        F  = 1./(wc'+ws);
        
        else %3D
        Gamma=Gamma+2*MembLambda*(Delta1^(-2));  
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
         
        ac  = (1:N)'/(N+1)*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = (1:M)'/(M+1)*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as)); 
        
            
        acz  = (1:K)'/(K+1)*pi;
        wcz  = 2*MembLambda*(Delta1^(-2))*(2-cos(acz));
             
            
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
        Alpha= BendLambda*(Delta2^(-4));   
        Gamma= 6*BendLambda*(Delta1^(-4))+Dispabs;

        if isvector(F) % 1D
        
        ac  = (1:N)'/(N+1)*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2); 
        F  = 1./(wc);
        
        elseif ismatrix(F) %2D 
        
        ac  = (1:N)'/(N+1)*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = (1:M)'/(M+1)*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);   
        
        F  = 1./(wc'+ws);    
                
             
        else %3D
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
         Gamma= Gamma+6*BendLambda*(Delta3^(-4))+Dispabs;
         Beta=  4*BendLambda*(Delta1^(-2))*(Delta2^(-2))+4*BendLambda*(Delta3^(-2))*(Delta2^(-2));
         Xe=4*BendLambda*(Delta2^(-4))+4*BendLambda*(Delta2^(-2)*Delta3^(-2))-4*BendLambda*(Delta1^(-4))-4*BendLambda*(Delta1^(-2)*Delta3^(-2));
        ac  = (1:N)'/(N+1)*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = (1:M)'/(M+1)*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);
         
        acz  = (1:K)'/(K+1)*pi;
        acxz=2*(2*((cos(acz)).^2))-1;
        acyz=2*cos(2*acz);
        acxyz=2*cos(acz);
        acz  = 2*(1-cos(acz));
        Teta=  2*BendLambda*(Delta3^(-2))*(Delta2^(-2));
        Eta= (-1)*BendLambda*(Delta2^(-4))+BendLambda*(Delta3^(-4));
        wcz  = Xe*acxyz+Eta*acyz+Teta*acxz+Beta*acz+Alpha*acz.^2;  
        
        
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end         
            
              
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
        Alpha1= (2*ElasticMu+ElasticLambda)* (Delta1^(-2));  
        Beta1=   ElasticMu*(Delta2^(-2));  
        Gamma1=  (-1)*(2*ElasticMu+ElasticLambda)* (Delta1^(-2));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta1^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta1^(-2));  
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2));
        
        
        if isvector(F) % 1D
            
        if size(F,2)==1
        NN=size(F,1);    
        ac  =(1:NN)'/(NN+1)*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma1*acx+Beta1*ac+Alpha1*ac; 
        F  = 1./(wc);    
        end
        
        if size(F,1)==1 
        NN=size(F,2);    
        ac  = (1:NN)'/(NN+1)*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac; 
        F  = 1./(wc);    
        end
        
        elseif ismatrix(F) %2D 
       
        ac  = (1:N)'/(N+1)*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =(1:M)'/(M+1)*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy; 
            
        F  = 1./(wc'+ws);   
              
        else %3D
            
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1);
         
        ac  = (1:N)'/(N+1)*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta3^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2))+ElasticMu*(Delta3^(-2));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =(1:M)'/(M+1)*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        Beta1=   ElasticMu*(Delta2^(-2))+ElasticMu*(Delta3^(-2));
        Gamma1=  Gamma1+ElasticMu*(Delta3^(-2));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy;    
            
        acz  = (1:K)'/(K+1)*pi;
        acx=2*cos(acz);
        acy=2*(2*((cos(acz)).^2))-1;
        acz  = 2*(1-cos(acz));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta3^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta3^(-2));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta1^(-2));
        Gamma2=  ElasticMu*(Delta3^(-2))+ElasticMu*(Delta1^(-2));
        wcz  = Gamma2*acx+Beta2*acz+Alpha3*acz+Alpha2*acy;    
            
        lambda = kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
                 
        end    
        
    
        else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
       end   
        case 3   % Sliding
        if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
           

        %Parameter of the absolute displacement penalty plus Gamma
        Gamma=2*MembLambda*(Delta2^(-2))+Dispabs; 
        if isvector(F) % 1D
            
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));  
        F  = 1./(wc);

        
        elseif ismatrix(F) %2D
        
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = (1:M)'/(M+1)*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as));
        F  = 1./(wc'+ws);
        
        else %3D
        Gamma=Gamma+2*MembLambda*(Delta1^(-2));  
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
         
        ac  = ((1:N)'-1)/N*pi;
        wc  = 2*MembLambda*(Delta1^(-2))*(2-cos(ac));

        as  = (1:M)'/(M+1)*pi;
        ws  = 2*MembLambda*(Delta1^(-2))*(2-cos(as)); 
        
            
        acz  = ((1:K)'-1)/K*pi;
        wcz  = 2*MembLambda*(Delta1^(-2))*(2-cos(acz));
             
            
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
        Alpha= BendLambda*(Delta2^(-4));    
        Gamma= 6*BendLambda*(Delta1^(-4))+Dispabs;

        if isvector(F) % 1D
        
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2); 
        F  = 1./(wc);
        
        elseif ismatrix(F) %2D 
        
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = (1:M)'/(M+1)*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);   
        
        F  = 1./(wc'+ws);              
             
        else %3D
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1); 
         Gamma= Gamma+6*BendLambda*(Delta3^(-4))+Dispabs;
         Beta=  4*BendLambda*(Delta1^(-2))*(Delta2^(-2))+4*BendLambda*(Delta3^(-2))*(Delta2^(-2));
         Xe=4*BendLambda*(Delta2^(-4))+4*BendLambda*(Delta2^(-2)*Delta3^(-2))-4*BendLambda*(Delta1^(-4))-4*BendLambda*(Delta1^(-2)*Delta3^(-2));
        ac  = ((1:N)'-1)/N*pi;
        acx=2*(2*((cos(ac)).^2))-1; 
        wc  = 2*BendLambda*(Delta1^(-4))*acx+BendLambda*(Delta1^(-4))*(14-8*(cos(ac))+(2-2*cos(ac)).^2);

        as  = (1:M)'/(M+1)*pi;
        asx=2*(2*((cos(as)).^2))-1; 
        ws  = 2*BendLambda*(Delta1^(-4))*asx+BendLambda*(Delta1^(-4))*(14-8*(cos(as))+(2-2*cos(as)).^2);
         
        acz  = ((1:K)'-1)/K*pi;
        acxz=2*(2*((cos(acz)).^2))-1;
        acyz=2*cos(2*acz);
        acxyz=2*cos(acz);
        acz  = 2*(1-cos(acz));
        Teta=  2*BendLambda*(Delta3^(-2))*(Delta2^(-2));
        Eta= (-1)*BendLambda*(Delta2^(-4))+BendLambda*(Delta3^(-4));
        wcz  = Xe*acxyz+Eta*acyz+Teta*acxz+Beta*acz+Alpha*acz.^2;  
        
        
        lambda = Gamma+kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
        end         
            
              
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
        Alpha1= (2*ElasticMu+ElasticLambda)* (Delta1^(-2));  
        Beta1=   ElasticMu*(Delta2^(-2));  
        Gamma1=  (-1)*(2*ElasticMu+ElasticLambda)* (Delta1^(-2));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta1^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta1^(-2));  
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2));
        
        
        if isvector(F) % 1D
            
        if size(F,2)==1
        NN=size(F,1);    
        ac  =(1:NN)'/(NN+1)*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma1*acx+Beta1*ac+Alpha1*ac; 
        F  = 1./(wc);    
        end
        
        if size(F,1)==1 
        NN=size(F,2);    
        ac  = ((1:NN)'-1)/NN*pi;
        acx=2*cos(ac);
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac; 
        F  = 1./(wc);    
        end
        
        elseif ismatrix(F) %2D 
       
        ac  = ((1:N)'-1)/N*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =(1:M)'/(M+1)*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy; 
            
        F  = 1./(wc'+ws);   
              
        else %3D
            
         e1 = ones(N,1);
         e2 = ones(M,1); 
         e3 = ones(K,1);
         
        ac  = ((1:N)'-1)/N*pi;
        acx=2*cos(ac);
        acy=2*(2*((cos(ac)).^2))-1;
        ac  = 2*(1-cos(ac));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta3^(-2)); 
        Gamma2=  ElasticMu*(Delta1^(-2))+ElasticMu*(Delta3^(-2));
        wc  = Gamma2*acx+Beta2*ac+Alpha3*ac+Alpha2*acy;    
            
            
        as  =(1:M)'/(M+1)*pi;
        asx=2*cos(as);
        asy=2*(2*((cos(as)).^2))-1;
        as  = 2*(1-cos(as));
        Beta1=   ElasticMu*(Delta2^(-2))+ElasticMu*(Delta3^(-2));
        Gamma1=  Gamma1+ElasticMu*(Delta3^(-2));
        ws  = Gamma1*asx+Beta1*as+Alpha1*as+Alpha2*asy;    
            
        acz  = ((1:K)'-1)/K*pi;
        acx=2*cos(acz);
        acy=2*(2*((cos(acz)).^2))-1;
        acz  = 2*(1-cos(acz));
        Alpha2=  ((ElasticMu+ElasticLambda)/4)* (Delta3^(-1))* (Delta2^(-1));
        Alpha3=  ElasticMu* (Delta3^(-2));
        Beta2=  (2*ElasticMu+ElasticLambda)* (Delta2^(-2))+ElasticMu*(Delta1^(-2));
        Gamma2=  ElasticMu*(Delta3^(-2))+ElasticMu*(Delta1^(-2));
        wcz  = Gamma2*acx+Beta2*acz+Alpha3*acz+Alpha2*acy;    
            
        lambda = kron(e3,kron(e2,wc)) + kron(e3,kron(ws,e1))...
            + kron(wcz,kron(e2,e1));    
        matrix3D=zeros(N, M, K);
          for k=1:K
           for j=1:M 
              for i=1:N   
               matrix3D (i,j,k) = lambda( (k-1)*(N*M) + (j-1)*N + i );
              end
           end
          end    
            F  = 1./(matrix3D);    
                 
        end    
        
    
        else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
       end   

           otherwise
            error('Only Circulant(0), Neumann(1), Dirichlet(2) and Sliding(3) Boundaries are supported');
       end    
     
      
        sm = prod(size(F));
        if nargout >=2
            ld = log(F);
            if prm(4)==0, ld(1,1,1) = 0; end
            ld = -sum(ld(:));
        end
        if prm(4)==0
            F(1,1,1) = 0;
            sm       = sm - 1;
        end;
        if nargout >=2
           ld = 3*ld + sm*sum(2*log(prm(1:3)));
        end
    else
        for j=1:size(F,5),
            for i=1:size(F,4),
                % The differential operator is symmetric, so the Fourier transform should be real
                F(:,:,:,i,j) = real(fftn(F(:,:,:,i,j)));
            end
        end
        ld = 0;
        sm = 0;
        for k=1:size(F,3),
            % Compare the following with inverting a 3x3 matrix...
            A   = F(:,:,k,:,:);
            dt  = A(:,:,:,1,1).*(A(:,:,:,2,2).*A(:,:,:,3,3) - A(:,:,:,2,3).*A(:,:,:,3,2)) +...
                  A(:,:,:,1,2).*(A(:,:,:,2,3).*A(:,:,:,3,1) - A(:,:,:,2,1).*A(:,:,:,3,3)) +...
                  A(:,:,:,1,3).*(A(:,:,:,2,1).*A(:,:,:,3,2) - A(:,:,:,2,2).*A(:,:,:,3,1));
            msk     = dt<=0;
            if prm(4)==0 && k==1, msk(1,1,1) = true; end;
            dt      = 1./dt;
            dt(msk) = 0;
            if nargout>=2
                sm      = sm + sum(sum(~msk));
                ld      = ld - sum(log(dt(~msk)));
            end
            F(:,:,k,1,1) = (A(:,:,:,2,2).*A(:,:,:,3,3) - A(:,:,:,2,3).*A(:,:,:,3,2)).*dt;
            F(:,:,k,2,1) = (A(:,:,:,2,3).*A(:,:,:,3,1) - A(:,:,:,2,1).*A(:,:,:,3,3)).*dt;
            F(:,:,k,3,1) = (A(:,:,:,2,1).*A(:,:,:,3,2) - A(:,:,:,2,2).*A(:,:,:,3,1)).*dt;

            F(:,:,k,1,2) = (A(:,:,:,1,3).*A(:,:,:,3,2) - A(:,:,:,1,2).*A(:,:,:,3,3)).*dt;
            F(:,:,k,2,2) = (A(:,:,:,1,1).*A(:,:,:,3,3) - A(:,:,:,1,3).*A(:,:,:,3,1)).*dt;
            F(:,:,k,3,2) = (A(:,:,:,1,2).*A(:,:,:,3,1) - A(:,:,:,1,1).*A(:,:,:,3,2)).*dt;

            F(:,:,k,1,3) = (A(:,:,:,1,2).*A(:,:,:,2,3) - A(:,:,:,1,3).*A(:,:,:,2,2)).*dt;
            F(:,:,k,2,3) = (A(:,:,:,1,3).*A(:,:,:,2,1) - A(:,:,:,1,1).*A(:,:,:,2,3)).*dt;
            F(:,:,k,3,3) = (A(:,:,:,1,1).*A(:,:,:,2,2) - A(:,:,:,1,2).*A(:,:,:,2,1)).*dt;
        end
    end
    varargout{1} = F;
    if nargout>=2
        varargout{2} = [ld, sm];
    end

  else
    % Convolve with the Green's function via Fourier methods
    m = varargin{1};
    F = varargin{2};
%     if nargin < 4
%         bnd = 0;
%     else
%         bnd = varargin{4};
%     end
    v = zeros(size(m),'single');
    if size(F,4) == 1,
        % Simple case where convolution is done one field at a time
       
       %prm = varargin{3};
       
       for i=1:3,
            switch  bnd 
       case 0   % Circulant
           
            v(:,:,:,i) = ifftn(F.*fftn(m(:,:,:,i))*prm(i)^2,'symmetric');
            
       case 1   %  Neumann
       if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
         if isvector(F)
             
             if (gpuDeviceCount==0)
                 
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             
             end
             
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end
             
         elseif ismatrix(F)
             
             if (gpuDeviceCount==0)
             
             v(:,:,:,i) =   idct(idct(prm(i)^2*dct(dct(m(:,:,:,i)),[],2).*F),[],2);
             
             end
             
             if (gpuDeviceCount~=0)
                  
               v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row')),'cosine', 'inverse', 'two', 'column' ); 

             end
                
             
         else
              if (gpuDeviceCount==0)
                  
              v(:,:,:,i) =  idctn(idctn(idctn(dctn(dctn(dctn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
              
              end
              
              if (gpuDeviceCount~=0)
              
              v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row')),'cosine', 'direct', 'two', 'third').*F),'cosine', 'inverse', 'two', 'row')),'cosine', 'inverse', 'two', 'column' )),'cosine', 'inverse', 'two', 'third'); 
              end
             
         end    
           
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
           
         if isvector(F)
             
             if (gpuDeviceCount==0)
                 
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             
             end
             
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end
             
         elseif ismatrix(F)
             
             if (gpuDeviceCount==0)
             
             v(:,:,:,i) =   idct(idct(prm(i)^2*dct(dct(m(:,:,:,i)),[],2).*F),[],2);
             
             end
             
             if (gpuDeviceCount~=0) 
                  
               v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row')),'cosine', 'inverse', 'two', 'column' ); 

             end
                
             
         else
              if (gpuDeviceCount==0)
                  
              v(:,:,:,i) =  idctn(idctn(idctn(dctn(dctn(dctn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
              
              end
              
              if (gpuDeviceCount~=0)
              
              v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row')),'cosine', 'direct', 'two', 'third').*F),'cosine', 'inverse', 'two', 'row')),'cosine', 'inverse', 'two', 'column' )),'cosine', 'inverse', 'two', 'third'); 
             
              end
             
         end 
        
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
           
         if isvector(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             end
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end
             
         elseif ismatrix(F)
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dct(dct(m(:,:,:,i)),[],2);
            end
            if (gpuDeviceCount~=0)
            m(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row');
            end
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0) 
            v(:,:,:,j) =  idct(idct(a),[],2);
            end
            if (gpuDeviceCount~=0)
            v(:,:,:,j) =Discrete_Transform((Discrete_Transform(a, 'cosine', 'inverse', 'two', 'column')),'cosine', 'inverse', 'two', 'row');
            end
         end
                
         else
             
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dctn(dctn(dctn(m(:,:,:,i)),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
             m(:,:,:,i) = Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'row')),'cosine', 'direct', 'two', 'third'); 
            
            end
            
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0)
            v(:,:,:,j) =  idctn(idctn(idctn(a),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
                
            v(:,:,:,j) = Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(a, 'cosine', 'inverse', 'two', 'column')),'cosine', 'inverse', 'two', 'row')),'cosine', 'inverse', 'two', 'third'); 
               
            end
            
         end 
              
         end
         
       else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
        
       end
       
       case 2   % Dirichlet
       if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
         if isvector(F)
             
             if (gpuDeviceCount==0)
                 
             v(:,:,:,i) = idst(dst(m(:,:,:,i)).*F);
             
             end
             
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column').*F),'sine', 'inverse', 'one', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row').*F),'sine', 'inverse', 'one', 'row');
             end
             end
             
         elseif ismatrix(F)
             
             if (gpuDeviceCount==0)
             
             v(:,:,:,i) =   idstn(idstn(prm(i)^2*dstn(dstn(m(:,:,:,i)),[],2).*F),[],2); 
             
             end
             
             if (gpuDeviceCount~=0)
                  
               v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row').*F),'sine', 'inverse', 'one', 'row')),'sine', 'inverse', 'one', 'column' ); 

             end
                
             
         else
              if (gpuDeviceCount==0)
                  
              v(:,:,:,i) =  idstn(idstn(idstn(dstn(dstn(dstn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
              end
              
              if (gpuDeviceCount~=0)
              
              v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row')),'sine', 'direct', 'one', 'third').*F),'sine', 'inverse', 'one', 'row')),'sine', 'inverse', 'one', 'column' )),'sine', 'inverse', 'one', 'third'); 
              end
             
         end    
           
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
           
         if isvector(F)
             
             if (gpuDeviceCount==0)
                 
             v(:,:,:,i) = idst(dst(m(:,:,:,i)).*F);
             
             end
             
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column').*F),'sine', 'inverse', 'one', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row').*F),'sine', 'inverse', 'one', 'row');
             end
             end
             
         elseif ismatrix(F)
             
             if (gpuDeviceCount==0)
             
             v(:,:,:,i) =   idstn(idstn(prm(i)^2*dstn(dstn(m(:,:,:,i)),[],2).*F),[],2);
             end
             
             if (gpuDeviceCount~=0)
                 
               v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row').*F),'sine', 'inverse', 'one', 'row')),'sine', 'inverse', 'one', 'column' ); 

             end
                
             
         else
              if (gpuDeviceCount==0)
                  
              v(:,:,:,i) =  idstn(idstn(idstn(dstn(dstn(dstn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
             
              end
              
              if (gpuDeviceCount~=0)
              
              v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row')),'sine', 'direct', 'one', 'third').*F),'sine', 'inverse', 'one', 'row')),'sine', 'inverse', 'one', 'column' )),'sine', 'inverse', 'one', 'third'); 
             
              end
             
         end 
        
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
           
         if isvector(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) = idstn(dstn(m(:,:,:,i)).*F);
             end
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column').*F),'sine', 'inverse', 'one', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row').*F),'sine', 'inverse', 'one', 'row');
             end
             end
             
         elseif ismatrix(F)
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dstn(dstn(m(:,:,:,i)),[],2);
            end
            if (gpuDeviceCount~=0)
            m(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row');
            end
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0) 
            v(:,:,:,j) =  idstn(idstn(a),[],2);
            end
            if (gpuDeviceCount~=0)
            v(:,:,:,j) =Discrete_Transform((Discrete_Transform(a, 'sine', 'inverse', 'one', 'column')),'sine', 'inverse', 'one', 'row');
            end
         end
                
         else
             
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dstn(dstn(dstn(m(:,:,:,i)),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
             m(:,:,:,i) = Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'column')),'sine', 'direct', 'one', 'row')),'sine', 'direct', 'one', 'third'); 
            
            end
            
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0)
            v(:,:,:,j) =  idstn(idstn(idstn(a),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
                
            v(:,:,:,j) = Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(a, 'sine', 'inverse', 'one', 'column')),'sine', 'inverse', 'one', 'row')),'sine', 'inverse', 'one', 'third'); 
               
            end
            
         end 
              
         end
         
       else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
        
       end
	   
       case 3   % Sliding 
         if (MembLambda~=0 && BendLambda==0 && ElasticMu==0 && ElasticLambda==0 )  % Membrane energy model
         
         if isvector(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             end
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end   
             
         elseif ismatrix(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) =  idct(idst(prm(i)^2*dct(dst(m(:,:,:,i)),[],2).*F),[],2);
             end
             if (gpuDeviceCount~=0)    
              v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column').*F),'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column' ); 
             end
             
         else
             if (gpuDeviceCount==0)
              v(:,:,:,i) =  idctn(idctn(idstn(prm(i)^2*dctn(dctn(dstn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
             end
             if (gpuDeviceCount~=0)
             v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'third').*F),'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column' )),'cosine', 'inverse', 'two', 'third' ); 
             end
         end    
           
        
       elseif (MembLambda==0 && BendLambda~=0 && ElasticMu==0 && ElasticLambda==0)  % Bending energy model
           
         if isvector(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             end
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end   
             
         elseif ismatrix(F)
             if (gpuDeviceCount==0)
             v(:,:,:,i) =  idct(idst(prm(i)^2*dct(dst(m(:,:,:,i)),[],2).*F),[],2);
             end
             if (gpuDeviceCount~=0)
                
              v(:,:,:,i) =Discrete_Transform((Discrete_Transform((prm(i)^2*Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column').*F),'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column' ); 
             end
             
         else
             if (gpuDeviceCount==0)
              v(:,:,:,i) =  idctn(idctn(idstn(prm(i)^2*dctn(dctn(dstn(m(:,:,:,i)),[],2),[],3).*F),[],2), [],3);
             end
             if (gpuDeviceCount~=0)
             v(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'third').*F),'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column' )),'cosine', 'inverse', 'two', 'third' ); 
             end
         end
        
       elseif (MembLambda==0 && BendLambda==0 && ElasticMu~=0 && ElasticLambda~=0)  % Linear elastic energy model
           
         if isvector(F)
             
             if (gpuDeviceCount==0)
             v(:,:,:,i) = idct(dct(m(:,:,:,i)).*F);
             end
             if (gpuDeviceCount~=0)
             if size(F,2)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'column').*F),'cosine', 'inverse', 'two', 'column');
             end
             if size(F,1)==1    
             v(:,:,:,i) =Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'cosine', 'direct', 'two', 'row').*F),'cosine', 'inverse', 'two', 'row');
             end
             end 
             
         elseif ismatrix(F)
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dct(dst(m(:,:,:,i)),[],2);
            end
            if (gpuDeviceCount~=0)
            m(:,:,:,i) = Discrete_Transform((Discrete_Transform(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column'); 
             end
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0) 
            v(:,:,:,j) =  idct(idst(a),[],2);
            end
            if (gpuDeviceCount~=0)
            v(:,:,:,j) = Discrete_Transform((Discrete_Transform(a, 'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column'); 
             end
         end
                
         else
             
         for i=1:3,
            if (gpuDeviceCount==0) 
            m(:,:,:,i) = dctn(dctn(dstn(m(:,:,:,i)),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
            m(:,:,:,i) =Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(m(:,:,:,i), 'sine', 'direct', 'one', 'row')),'cosine', 'direct', 'two', 'column')),'cosine', 'direct', 'two', 'third'); 
             end
         end
         for j=1:3,
            a = single(0);
            for i=1:3,
                a = a + F(:,:,:,j,i).*m(:,:,:,i)*prm(i)^2;
            end
            if (gpuDeviceCount==0)
            v(:,:,:,j) =  idctn(idctn(idstn(a),[],2),[],3);
            end
            if (gpuDeviceCount~=0)
            v(:,:,:,j) = Discrete_Transform_3D((Discrete_Transform_3D((Discrete_Transform_3D(a, 'sine', 'inverse', 'one', 'row')),'cosine', 'inverse', 'two', 'column')),'cosine', 'inverse', 'two', 'third'); 
             end
         end 
              
         end
         
       else
               error('Only Membrane, Bending and Linear elastic energy models are supported');
        
       end

	   
        otherwise
            error('Only Circulant(0), Neumann(1), Dirichlet(2) and Sliding(3) Boundaries are supported');   
            
            end
        end 
       
    else
        % More complicated case for dealing with linear elasticity, where
        % convolution is not done one field at a time
        for i=1:3,
            m(:,:,:,i) = fftn(m(:,:,:,i));
        end
        for j=1:3,
            a = single(0);
            for i=1:3,
                a = a + F(:,:,:,j,i).*m(:,:,:,i);
            end
            v(:,:,:,j) = ifftn(a,'symmetric');
        end
    end
    varargout{1} = v;

end
%__________________________________________________________________________________
