classdef sigmaplotter < handle
    %A class which implements visualization of conductivity distributions.
    %Add new methods here for different kinds of plots.

    properties
        Mesh %first order mesh structure
        fh   %figure handle
        cmp %colormap
    end
    methods
        function obj = sigmaplotter(Mesh,fh,cmp)
            obj.Mesh = Mesh;
            obj.cmp = cmp;
            handle1 = figure(fh(1));
            set(handle1,'Units','normalized','OuterPosition',[0 0.6 0.3 0.4])
            if length(fh) == 2
                handle2 = figure(fh(2));
                set(handle2,'Units','normalized','OuterPosition',[0 0.2 0.3 0.4])
                obj.fh = [handle1 handle2];
            else
                obj.fh = handle1;
            end
        end

        function basicplot(self,sigma,str)
                 basic2Dplot(self,sigma,str)
        end

        %2D plotter function - give sigma as a vector
        function basic2Dplot(self,sigma,str)
            if length(self.fh) == 2
                ng = 0.5*length(sigma);
                delta_sigma = sigma(ng+1:end);
                sigma = sigma(1:ng);
            end

            figure(self.fh(1))
            clf
            self.PlotSolution(self.Mesh.g,self.Mesh.H,sigma,self.fh(1)),axis image, colormap(self.cmp), colorbar
            if exist('str','var') && ~isempty(str)
                title(str{1})
            end

            if length(self.fh) == 2
                figure(self.fh(2))
                clf
                self.PlotSolution(self.Mesh.g,self.Mesh.H,delta_sigma,self.fh(2)),axis image, colormap(self.cmp), colorbar
                if exist('str','var') && ~isempty(str)
                    title(str{2})
                end
            end
        end

        function PlotSolution(~, g,H,s, fighandle, bndr)
            
            if ~exist('fighandle','var')
                fighandle = figure('Name', mfilename);
            end
            if size(g,2) < 3
                %z = zeros(size(g(:,1)));  
                z = s;
            else
                z = s;
            end
            if(isvalid(fighandle))
                set(groot,'CurrentFigure', fighandle ), clf 
                h = trisurf(H, g(:,1), g(:,2), z, s);
                view(2)
                h.EdgeColor = 'none';
                h.FaceColor = 'interp';
                grid off
                colorbar;
                if exist('bndr','var')
                    PlotBoundary(g,bndr);
                end
            end
        end
    end
end