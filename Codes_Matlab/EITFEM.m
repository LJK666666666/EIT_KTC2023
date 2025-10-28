classdef EITFEM < handle
    %Forward solver class for EIT.

    properties
        %THE FIRST ORDER MESH IS NOT USED IN KTC VERSION OF THE CODES. IT
        %IS INSTEAD ALWAYS CONSTRUCTED FROM THE SECOND ORDER MESH.
        %Mesh    %A first order mesh structure which contains the following fields:
        %g       Forward mesh node coordinates, ng x 2 array (triangles) (1st order, i.e. only vertices of the triangles)
        %H       Forward mesh elements, nH x 3 array (triangles), indices refer to rows in g
        %elfaces Electrode surface elements, Nel x 1 cell array, each cell n x 2 (lines), indices refer to rows in g
        %Node    %Node structure
        %Element %Element structure

        Mesh2     %A second order mesh structure which contains the following fields:
        %g        Second order mesh node coordinates, ng2 x 2 array (triangles)
        %H        Second order mesh elements 
        %elfaces  Second order mesh electrode surface elements (similar to elfaces but list 2nd order nodes too)
        %Node     %2nd order node structure
        %Element  %2nd order element structure

        Nel     %Number of electrodes
        nH      %Number of elements in the first order forward mesh
        ng      %Number of nodes in the first order forward mesh
        nH2     %Number of elements in the second order forward mesh
        ng2     %Number of nodes in the second order forward mesh

        sigmamin %Minimum allowed conductivity value
        sigmamax  %Maximum allowed conductivity value
        zmin %minimum allowed contact impedance value

        C       %C matrix contains the basis function of electrode potentials (Nel x Nel-1 array)
                %these are the basis vectors of the electrode potentials
        vincl   %A logical array (Nmeas x 1) indicating which measurements will actually be used in inversion

        %Forward solution data is stored in the following fields.
        Imeas   %Electrode currents
        Umeas   %Voltages between electrodes

        Mpat    %Measurement pattern, encoding how the voltages between electrodes are measured
        Inj     %Excitation pattern, encoding how current is injected between electrodes

        A       %FEM matrix
        b       %Right-hand-side of FEM equation A*theta = b
        theta   %Plain FEM solution vector (potential field and coefficients of electrode potential basis vectors)
        Pot     %the potential field
        QC      %Matrix used to extract measurements from theta (i.e. [0 C])

        Ai      %The index decomposition (rows) used in caluclating the Jacobian
        Aj      %Similar to above (columns)
        Av      %Linear basis function gradient dot product integral values used in calculating the Jacobian

        InvGamma_n %inverse of covariance matrix Gamma of the measurement noise
        Ln         %chol(InvGamma_n)
        %the above is used to evaluate the Least Squares data residual when
        %needed
          
    end

    methods
        
        function obj = EITFEM(~, Mesh2, Inj, Mpat, vincl, sigmamin)
            %The class constructor.
            %the 1st order mesh is not needed as input (argument #1), as
            %the 1st order mesh will be constructed from the 2nd order mesh

            %set properties according to the input
            %obj.Mesh = Mesh;
            obj.Mesh2 = Mesh2;
            obj.Inj = Inj;
            if exist('Mpat','var') %Measurement pattern only used in current excitation mode
                obj.Mpat = Mpat;
            end  
            if exist('sigmamin','var') %Minimum allowed conductivity value
                if ~isempty(sigmamin)
                    obj.sigmamin = sigmamin;
                else
                    obj.sigmamin = 1e-9;
                end
            else
                obj.sigmamin = 1e-9;
            end            
            if exist('sigmmax','var') %Maximum allowed conductivity value
                obj.sigmamax = sigmamax;
            else
                obj.sigmamax = 1e9;
            end          
            obj.zmin = 1e-6;
            if exist('vincl','var')  
                obj.vincl = vincl;
            else
                obj.vincl = true(obj.Nel*obj.Nel,1);
            end
       
            %setting properties: add Node and Element structure
            %construction if not given?

            %set some properties derived from the input
            obj.Nel = length(Mesh2.elfaces);
            %obj.nH = size(Mesh.H,1);
            %obj.ng = size(Mesh.g,1);
            obj.nH2 = size(Mesh2.H,1);
            obj.ng2 = size(Mesh2.g,1);

            obj.C = [ones(1,obj.Nel-1);-eye(obj.Nel-1)]; %basis vectors for electrode potentials

        end

        function fsol = SolveForward(self, sigma, z)
            %Forward solution function - solves the potential field and
            %measured voltages given conductivity sigma and contact impedances
            %z
            
            %project negative / too small sigma values to a given minimum
            %project too high sigma values to a given maximum
            sigma(sigma < self.sigmamin) = self.sigmamin;
            sigma(sigma > self.sigmamax) = self.sigmamax;            
            z(z < self.zmin) = self.zmin;
            
            %Compute the conductivity-dependent part of the FEM matrix
            gN=max(size(self.Mesh2.Node));
            HN=max(size(self.Mesh2.H)); %number of elements
            k = 1;
            Arow = zeros(6*HN,6);
            Acol = zeros(6*HN,6);
            Aval = zeros(6*HN,6);
            for ii=1:HN % Go through all triangles    
                ind = self.Mesh2.H(ii,:);
                gg = self.Mesh2.g(ind,:);
                ss = sigma(ind([1,3,5]));
                int = self.grinprodgausquadnode(gg,ss);
                Acol(k:k+5,:) = repmat(ind, 6, 1);
                ind=ind.';
                Arow(k:k+5,:) = repmat(ind, 1, 6);
                Aval(k:k+5,:) = int;
                k = k + 6;
            end
            A0 = sparse(Arow,Acol,Aval,self.ng2+self.Nel-1,self.ng2+self.Nel-1);
            
            %Compute the rest of the FEM matrix
            M=sparse(zeros(gN,self.Nel));
            K=sparse(zeros(gN,gN));
            s=zeros(self.Nel,1);
            g=reshape([self.Mesh2.Node.Coordinate],2,gN)'; %Nodes
            for ii=1:HN
                % Go through all triangles
                ind=(self.Mesh2.Element(ii).Topology); % The indices to g of the ii'th triangle.
                if ~isempty(self.Mesh2.Element(ii).Electrode)  %Checks if the triangle ii is under an electrode
                    Ind=self.Mesh2.Element(ii).Electrode{2};
                    a=g(Ind(1),:);
                    b=g(Ind(2),:); %the 2nd order node
                    c=g(Ind(3),:);
                    InE=self.Mesh2.Element(ii).Electrode{1};         %Electrode index.
                    s(InE)=s(InE)+1/z(InE)*self.electrlen([a;c]);% Assumes straight electrodes.
                    bb1=self.boundquad1([a;b;c]);
                    bb2=self.boundquad2([a;b;c]);
                    for il=1:6
                        eind=find(self.Mesh2.Element(ii).Topology(il)==self.Mesh2.Element(ii).Electrode{2});
                        if eind
                            M(ind(il),InE)=M(ind(il),InE)-1/z(InE)*bb1(eind);
                        end
                        for im=1:6
                            eind1=find(self.Mesh2.Element(ii).Topology(il)==self.Mesh2.Element(ii).Electrode{2});
                            eind2=find(self.Mesh2.Element(ii).Topology(im)==self.Mesh2.Element(ii).Electrode{2});
                            if eind1 & eind2
                                K(ind(il),ind(im))=K(ind(il),ind(im))+...
                                    1/z(InE)*bb2(eind1,eind2);
                            else
                            end
                        end
                    end
                end
            end
            S=sparse(diag(s));
            S=self.C'*S*self.C;
            M=M*self.C;
            S0=[K,M;M',S];
            
            self.A = A0 + S0;
            self.b=[zeros(self.ng2,size(self.Inj,2));self.C'*self.Inj]; %RHS vector
            UU=self.A\self.b; %solve the FEM system of equations
            self.theta = UU; %FEM solution vectors for each current injection
            self.Pot = UU(1:self.ng2,:); %the electric potential field for each current injection
            self.Imeas = self.Inj; %Injected currents
            self.Umeas = self.Mpat'*self.C*self.theta(self.ng2+1:end,:); %Measured voltages between electrodes
            self.Umeas = self.Umeas(self.vincl==1);
            self.QC = [zeros(size(self.Mpat,2),self.ng2) self.Mpat'*self.C];
            fsol = self.Umeas(:); %measured potentials, vectorized
            
        end

        function Js = Jacobian(self, sigma, z)
            %Computes the conductivity Jacobian.
            
            %If sigma and z are given as input, this function first computes the forward
            %solution using those values.
            %Otherwise (sigma and z empty), a forward solution saved in the class properties is used, and
            %an error is returned if it doesn't exist.
            
            %Also, this function calls GradientMatrix if the gradient data is not yet
            %saved in the object properties.
            
            
            if ~isempty(sigma) && ~isempty(z) %compute forward solution here
                self.SolveForward(sigma,z);
            else
                if isempty(self.theta)
                    error('Jacobian cannot be computed without first computing a forward solution')
                end
            end
            
            %If the gradient data has not been saved in object properties yet, compute
            %here
            if isempty(self.Ai)
                tempElements(length(self.Mesh2.Element)) = struct();
                for ii = 1:length(self.Mesh2.Element)
                    tempElements(ii).Topology = self.Mesh2.Element(ii).Topology([1,3,5]);
                    tempElements(ii).Electrode = self.Mesh2.Element(ii).Electrode;
                end
                Agrad = self.jacob_nodesigma2nd3(self.Mesh2.Node,tempElements,self.Mesh2.Node,self.Mesh2.Element);
                [self.Ai, self.Aj, self.Av] = self.regroupAgrad2cell(Agrad);
            end
            
            m = length(sigma); %number of conductivity parameters
            n = size(self.Umeas(:),1); %number of measurements in a data set
            
            Jleft  = -self.QC/self.A;
            Jright = self.theta;
            
            Js = zeros(n,m);
            for ii=1:m
                Jid = self.Ai{ii};
                Jtemp   = Jleft(:,Jid)*self.Av{ii}*Jright(Jid,:);
                Js(:,ii) = Jtemp(self.vincl==1);
            end

        end
        
        function Jz = Jacobianz(self, sigma, z)
            %Computes the contact impedance Jacobian.
            
            %If sigma and z are given as input, this function first computes the forward
            %solution using those values.
            %Otherwise (sigma and z empty), a forward solution saved in the class properties is used, and
            %an error is returned if it doesn't exist.
            
            if ~isempty(sigma) && ~isempty(z) %compute forward solution here
                self.SolveForward(sigma,z);
            else
                if isempty(self.theta)
                    error('Jacobian cannot be computed without first computing a forward solution')
                end
            end
            
            dA_dz = self.ComputedA_dz(self,z);
            
            m = length(z); %number of contact impedances
            n = size(self.Umeas(:),1); %number of measurements in a data set
            
            Jleft  = -self.QC/self.A;
            Jright = self.theta;
            
            Jz = zeros(n,m);
            for ii=1:m
                Jtemp   = Jleft*dA_dz{ii}*Jright;
                Jz(:,ii) = Jtemp(self.vincl==1);
            end

        end
        
        function SetInvGamma(self, noisepercentage, noisepercentage2, measdata)
            %Calculate and set the inverse of covariance matrix based on
            %the noise levels given as arguments.
            %Input:
            %       noisepercentage = noise level of the
            %       measurements. This is multiplied by the absolute value
            %       of each measurement to get the noise standard deviation of that
            %       measurement.
            
            %       noisepercentage2 (optional) = a constant noise level for all
            %       measurements. This coefficient is scaled to the
            %       difference of minimum and maximum measurement values to
            %       get the noise standard deviation.
            %
            %       measdata (optional) = measurement data which is used to
            %       compute the noise covariance matrix, if this input is
            %       not used a forward solution should be ran first so that
            %       simulated data is used to compute the noise covariance
            %
            %       The two types of noise above are added together.
            %NOTE: This does not add noise to the solution of the forward
            %problem! This just calculates the weighting matrix used in the
            %inverse problem.
            
            %Check the optional arguments:
            if nargin < 3 || isempty(noisepercentage2)
                noisepercentage2 = 0;
            end
            
            %Check if measurement data was provided
            if exist('measdata','var')
                meas = measdata;
            else %use previously simulated data
                meas = self.Umeas;
            end
            
            %Calculating the model variance of the noise
            %Noise level relative to the abs of measurement value:
            
            var_meas = ((noisepercentage/100)*(abs(meas))).^2;
            var_meas = var_meas + ((noisepercentage2/100)*(max(max(abs(meas)))))^2;
            
            Gamma_n = diag(var_meas(:));
            
            self.InvGamma_n = sparse(inv(Gamma_n));
            self.Ln = chol(self.InvGamma_n);%Store both invGamma and it's Cholesky
        end

        function int=grinprodgausquadnode(~, g,sigma)

            % The function int=grinprodgausquadnode(g,sigma);
            % calculates the gradient part in the linear 
            % FEM in EIT.
            
            % P. Ronkanen and M. Vauhkonen 10.5. 1996
            
            w=[1/6*ones(3,1)];
            ip=[1/2 0;1/2 1/2;0 1/2];
            
            int=0;
             for ii=1:3
              S=[1-ip(ii,1)-ip(ii,2);ip(ii,1);ip(ii,2)];
              L=[4*(ip(ii,1)+ip(ii,2))-3, -8*ip(ii,1)-4*ip(ii,2)+4, ...
                 4*ip(ii,1)-1, 4*ip(ii,2), 0, -4*ip(ii,2); ...
                 4*(ip(ii,1)+ip(ii,2))-3, -4*ip(ii,1), ...
                 0, 4*ip(ii,1), 4*ip(ii,2)-1, -8*ip(ii,2)-4*ip(ii,1)+4];
              Jt=L*g;
              iJt=inv(Jt);
              dJt=abs(det(Jt));
              G=iJt*L;
              int=int+w(ii)*(S'*sigma)*G'*G*dJt;
             end
            
        end
            
        function int=boundquad1(~, g)
            
            %boundquad1 calculates the boundary integral
            % of one quadratic basis function over the curve defined by the coordinates in g.
            % Function int=boundquad1(g,ind) calculates the boundary integral
            % of one quadratic basis function over the curve defined by the coordinates in g.
            %
            % INPUT
            %
            % g = integration points
            %
            % OUTPUT
            %
            % int = value of the integral
            
            % 10.5. 1996 P. Ronkanen and M. Vauhkonen
            % University of Kuopio, Department of Applied Physics, PO Box 1627,
            % FIN-70211 Kuopio, Finland, email: Marko.Vauhkonen@uku.fi
            
            
            w=[1/2,1/2];
            ip=[1/2-1/6*sqrt(3),1/2+1/6*sqrt(3)];
            int=0;
             for ii=1:2
              S=[2*ip(ii)^2-3*ip(ii)+1; ...
                 -4*ip(ii)^2+4*ip(ii); ...
                 2*ip(ii)^2-ip(ii)];
              dJt=sqrt((g(1,1)*(4*ip(ii)-3)+g(2,1)*(4-8*ip(ii))+g(3,1)*(4*ip(ii)-1))^2+ ...
                        (g(1,2)*(4*ip(ii)-3)+g(2,2)*(4-8*ip(ii))+g(3,2)*(4*ip(ii)-1))^2);
              int=int+w(ii)*S*dJt;
             end
            
            
        end
            
        function int=boundquad2(~, g)
            
            %boundquad2 Computes the boundary integral of the product of two quadratic basis functions over the curve defined by the coordinates in g.
            % Function int=boundquad2(g) calculates the boundary integral
            % of the product of two quadratic basis function over the curve defined by the coordinates in g.
            %
            % INPUT
            %
            % g = integration points
            %
            % OUTPUT
            %
            % int = value of the integral
            
            % 10.5. 1996 P. Ronkanen and M. Vauhkonen
            % University of Kuopio, Department of Applied Physics, PO Box 1627,
            % FIN-70211 Kuopio, Finland, email: Marko.Vauhkonen@uku.fi
            
            w=[5/18,8/18,5/18];
            ip=[1/2-1/10*sqrt(15),1/2,1/2+1/10*sqrt(15)];
            int=0;
             for ii=1:3
              S=[2*ip(ii)^2-3*ip(ii)+1; ...
                 -4*ip(ii)^2+4*ip(ii); ...
                 2*ip(ii)^2-ip(ii)];
              dJt=sqrt((g(1,1)*(4*ip(ii)-3)+g(2,1)*(4-8*ip(ii))+g(3,1)*(4*ip(ii)-1))^2+ ...
                        (g(1,2)*(4*ip(ii)-3)+g(2,2)*(4-8*ip(ii))+g(3,2)*(4*ip(ii)-1))^2);
              int=int+w(ii)*S*S'*dJt;
             end
            
        end
            
        function [int]=electrlen(self,g)
            
            % The function int=bound2_2(g) calculates the length of the electrode
            % from g(1,:) to g(2,:).
            
            % 10.5. 1996 P. Ronkanen and M. Vauhkonen

            dJt=sqrt((g(2,1)-g(1,1))^2+(g(2,2)-g(1,2))^2); 
            int=dJt;
        end

        function J=jacob_nodesigma2nd3(self, Nodesig,Elementsig,Node,Element)

            % Function J=jacob_nodesigma(Node,Element,Agrad,U,U0,style);
            % computes the Jacobian of the measured voltages with respect to the
            % nodal values of the conductivity. If only THe Node nad Element inormation is
            % given then the integral term is computed, which need to be computed
            % only once for each mesh.
            %
            % Node=node data
            % Nodesig=node data for sigma
            % Element=element data
            % Agrad=\int_{Element(ii) \nabla\phi_i\cdot\nabla\phi_j
            % U=voltages of the injected currents
            % U0=voltages of the measurement field
            
            % M. Vauhkonen 22.10.1999, Univeristy of Kuopio, Finland
            
            gNs=max(size(Nodesig));
            gN=max(size(Node));
            
            Agrad=sparse(gN^2,gNs); % Gradients of the basis functions integrated over
            % each element.
            
            for jj=1:gNs
                Aa=sparse(gN,gN);
                El=Nodesig(jj).ElementConnection;
                for ii=1:max(size(El))
                    indsig=Elementsig(El(ii)).Topology; % Indices of the element
                    ind=Element(El(ii)).Topology; % Indices of the element
                    gg=reshape([Node(ind).Coordinate],2,6)'; % A 3x2 matrix of triangle vertices in (x,y) coord.
                    I=find(jj==indsig);
                    if ~isempty(I)
                        anis=self.grinprodgaus2ndnode3(gg,I);
                        for il=1:6
                            for im=1:6
                                Aa(ind(il),ind(im))=Aa(ind(il),ind(im))+anis(il,im);
                            end
                        end
                    end
                end
                Agrad(:,jj)=Aa(:);
            end
            J=Agrad;
        end
            
        function int=grinprodgaus2ndnode3(~, g,I)
            
            % The function int=grinprodgausquadnode(g,sigma);
            % calculates the gradient part in the linear 
            % FEM in EIT.
            
            % P. Ronkanen and M. Vauhkonen 10.5. 1996
            % Fixed by J.J.
            
            w=[1/40*ones(3,1); 1/15*ones(3,1); 27/120];
            ip=[0   0; ...
                1   0; ...
                0   1; ...
                1/2 0; ...
                1/2 1/2; ...
                0   1/2; ...
                1/3 1/3 ];
            
            
            int=0;
             for ii=1:7
              S=[1-ip(ii,1)-ip(ii,2);ip(ii,1);ip(ii,2)];
              L=[4*(ip(ii,1)+ip(ii,2))-3, -8*ip(ii,1)-4*ip(ii,2)+4, ...
                 4*ip(ii,1)-1, 4*ip(ii,2), 0, -4*ip(ii,2); ...
                 4*(ip(ii,1)+ip(ii,2))-3, -4*ip(ii,1), ...
                 0, 4*ip(ii,1), 4*ip(ii,2)-1, -8*ip(ii,2)-4*ip(ii,1)+4];
              Jt=L*g;
              iJt=inv(Jt);
              dJt=abs(det(Jt));
              G=iJt*L;
              int=int+w(ii)*S(I)*G'*G*dJt;
             end
        end
            
        function [arow,acol,aval] = regroupAgrad2cell(~, Agrad,ng,ngs)
            % Regroups the gradient matrix Agrad for the computation of
            % the Jacobian in 2D/3D EIT. Regrouping tries to utilitize the full
            % advantage of the matrix sparsity.
            
            % K. Karhunen, 16.10.2006
            
            if nargin==2
            ng = sqrt(size(Agrad,1));  
            ngs = size(Agrad,2);
            end
            
            arow = cell(ng,1);
            acol = cell(ng,1);
            aval = cell(ng,1);
            for k=1:ngs
              S = reshape(Agrad(:,k),ng,ng); %this is top-left block of FEM matrix differentiated w.r.t. sigma_k
              [I,J] = find(S);
                
              I = unique(I);
              J = unique(J);
              
              arow{k} = I;
              acol{k} = J;
              aval{k} = S(I,J);
            end
        end

            function dA_dz = ComputedA_dz(~, self,z)
            dA_dz = cell(self.Nel,1);
            gN=max(size(self.Mesh2.Node));
            HN=max(size(self.Mesh2.H)); %number of elements
            M=sparse(zeros(gN,self.Nel));
            % K=sparse(zeros(gN,gN));
            K=cell(self.Nel,1);
            for ii=1:self.Nel
                K{ii} = sparse(zeros(gN,gN));
            end
            s=zeros(self.Nel,1);
            g=reshape([self.Mesh2.Node.Coordinate],2,gN)'; %Nodes
            for ii=1:HN
                % Go through all triangles
                ind=(self.Mesh2.Element(ii).Topology); % The indices to g of the ii'th triangle.
                if ~isempty(self.Mesh2.Element(ii).Electrode)  %Checks if the triangle ii is under an electrode
                    Ind=self.Mesh2.Element(ii).Electrode{2};
                    a=g(Ind(1),:);
                    b=g(Ind(2),:); %the 2nd order node
                    c=g(Ind(3),:);
                    InE=self.Mesh2.Element(ii).Electrode{1};         %Electrode index.
                    s(InE)=s(InE)-(1/(z(InE)^2))*self.electrlen([a;c]);% Assumes straight electrodes.
                    bb1=self.boundquad1([a;b;c]);
                    bb2=self.boundquad2([a;b;c]);
                    for il=1:6
                        eind=find(self.Mesh2.Element(ii).Topology(il)==self.Mesh2.Element(ii).Electrode{2});
                        if eind
                            M(ind(il),InE)=M(ind(il),InE)+(1/(z(InE)^2))*bb1(eind);
                        end
                        for im=1:6
                            eind1=find(self.Mesh2.Element(ii).Topology(il)==self.Mesh2.Element(ii).Electrode{2});
                            eind2=find(self.Mesh2.Element(ii).Topology(im)==self.Mesh2.Element(ii).Electrode{2});
                            if eind1 & eind2
                                K{InE}(ind(il),ind(im))=K{InE}(ind(il),ind(im))-...
                                    (1/(z(InE)^2))*bb2(eind1,eind2);
                            else
                            end
                        end
                    end
                end
            end
            for ii=1:self.Nel
                tmp = zeros(self.Nel,1);
                tmp(ii) = s(ii);
                S=sparse(diag(tmp));
                S=self.C'*S*self.C;
                Mtmp = zeros(size(M));
                Mtmp(:,ii) = M(:,ii);
                Mtmp=Mtmp*self.C;
                dA_dz{ii}=[K{ii},Mtmp;Mtmp',S];
            end
        end
          
    end
end