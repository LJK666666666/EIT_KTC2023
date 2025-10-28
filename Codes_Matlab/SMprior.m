classdef SMprior < handle
    
    properties
        mean %mean of the smoothness prior (can be vector or scalar)
        corrlength %used when forming the covariance matrix
        L %Cholesky factor of the inverse of the prior covariance (used to draw samples)
        %The inverse covariance itself is not included in the properties to
        %save space, but it can be computed as InvGamma = L'*L, if required
        c %small constant added to the diagonal of the prior covariance to ensure it's well-conditioned
        covariancetype %'Squared Distance' or 'Ornstein-Uhlenbeck'
    end
    
    methods
        %Class constructor
        function obj =  SMprior(ginv, corrlength, var, mean, covariancetype)
            obj.corrlength = corrlength;
            obj.mean = mean;
            obj.c = 1e-9; %default value
            if exist('covariancetype','var') && ~isempty(covariancetype)
                obj.covariancetype = covariancetype;
            else
                obj.covariancetype = 'Squared Distance'; %default
            end
            obj.ComputeL(ginv, corrlength, var);
        end
        
        %Cholesky factor of the inverse covariance
        function ComputeL(self, g, corrlength, var)
            ng = size(g,1);
            a = var - self.c;
            b = sqrt(-corrlength.^2./(2*log(.01)));
            Gamma_pr = zeros(ng,ng);
            for ii = 1:ng
                for jj = ii:ng
                    dist_ij = norm(g(ii,:)-g(jj,:));
                    if strcmp(self.covariancetype,'Squared Distance')
                        gamma_ij = a(1)*exp(-dist_ij^2/(2*b(1)^2));
                    elseif strcmp(self.covariancetype,'Ornstein-Uhlenbeck')
                        gamma_ij = a(1)*exp(-dist_ij/corrlength(1));
                    else
                        error('Unrecognized prior covariance type')
                    end
                    if ii == jj
                        gamma_ij = gamma_ij + self.c;
                    end
                    Gamma_pr(ii,jj) = gamma_ij;
                    Gamma_pr(jj,ii) = gamma_ij;
                end
            end        
            self.L = chol(inv(Gamma_pr));
        end
        
        %Random sample drawing function
        function samples = DrawSamples(self, nsamples)
            samples = self.mean + self.L\randn(size(self.L,1),nsamples);
        end
        
        %Least Squares residual function
        function res = EvalFun(self, args)
            sigma = args{1};
            res = 0.5*norm(self.L*(sigma - self.mean))^2;
        end
        
        %Hessian and gradient computing function
        function [Hess, grad] = ComputeHessAndGrad(self, args, nparam)
            sigma = args{1};
            Hess = self.L'*self.L;
            grad = Hess*(sigma-self.mean);

            if nparam > length(sigma)
                Hess = blkdiag(Hess,zeros(nparam-length(sigma)));
                grad = [grad; zeros(nparam-length(sigma),1)];
            end

        end
    end
    
end