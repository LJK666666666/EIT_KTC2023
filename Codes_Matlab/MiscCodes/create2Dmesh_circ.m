function [Mesh2,Mesh,elcenterangles] = create2Dmesh_circ(Nel,scaleparam,plotmesh,fignum)
R = 1; %circle radius
clscale = R/scaleparam;

filename = ['circ.geo'];
fname = 'circmesh';
fid = fopen(filename, 'w');
fprintf(fid, 'SetFactory("OpenCASCADE");\n\n\n');

elwidth = 360/(2*Nel);

% elcenterangles = [elwidth/2:2*elwidth:360]*(1/360)*2*pi;
elcenterangles = [0:2*elwidth:360]*(1/360)*2*pi; elcenterangles = elcenterangles(1:end-1);
elstartangles = elcenterangles-(0.5*elwidth)*(1/180)*pi;
gaplengths = diff(elstartangles) - (elwidth/180)*pi;
gaplengths = [gaplengths 2*pi-(elstartangles(end)+(elwidth/180)*pi)+elstartangles(1)];

[x,y] = pol2cart(elstartangles(1),R);
elstartp = [x,y];

if ~exist('plotmesh','var')
    plotmesh = 0;
end

fprintf(fid,['Point(' num2str(1) ') = {' num2str(elstartp(1)) ',' num2str(elstartp(2)) ',0};\n']);
for ii=1:2*Nel
    if ii < 2*Nel
        if mod(ii,2) == 1 %creating an electrode
            fprintf(fid,['Extrude {{0, 0, 1}, {0, 0, 0}, Pi/' num2str(Nel) '} {Point{' num2str(ii) '};}\n']);
        else %creating a gap between electrodes
            fprintf(fid,['Extrude {{0, 0, 1}, {0, 0, 0}, ' num2str( gaplengths(ii/2) ) '} {Point{' num2str(ii) '};}\n']);
        end
    else        
        fprintf(fid,['Extrude {{0, 0, 1}, {0, 0, 0}, ' num2str(0.95*gaplengths(end)) '} {Point{' num2str(ii) '};}\n']);
    end
end
%lastly, a line to connect the final point to the starting point 
fprintf(fid,['Line(' num2str(2*Nel+1) ') = {' num2str(2*Nel+1) ',1};\n']);
fprintf(fid,['Curve Loop(1) = {']);
for ii=1:2*Nel
    fprintf(fid,[num2str(ii) ', ']);
end
fprintf(fid,[num2str(num2str(2*Nel+1)) '};\n']);
fprintf(fid,['Plane Surface(1) = {1};\n']);

for ii=1:Nel
    fprintf(fid,['Physical Curve(' num2str(ii) ') = {' num2str((ii-1)*2+1) '};\n']);
end
fprintf(fid,['Physical Surface(' num2str(50) ') = {' num2str(1) '};\n']);
fprintf(fid,'Mesh.SecondOrderLinear = 1;\n'); %important - no curved edges

fclose(fid);
str = ['gmsh-4.10.3-Windows64\gmsh.exe ' filename ' -2 -order 2 -clscale ' num2str(clscale) ' -format m -o ' fname '.m'];
system(str);
eval(fname);

g2 = msh.POS(:,1:2);
H2 = msh.TRIANGLES6(:,1:6);
lns2 = msh.LINES3(:,1:3);

rmat = [cosd(90) -sind(90); sind(90) cosd(90)];
for ii=1:size(g2,1)
    g2(ii,:) = (rmat*g2(ii,:)')';
end

[g2,H2,lns2] = fixIndices2nd_2D(g2,H2,lns2);
%change format to be compatible with forward solvers
H2 = H2(:,[1,4,2,5,3,6]);
lns2 = lns2(:,[1,3,2]);

for ii=1:Nel
    tris = find(msh.LINES3(:,end) == ii);
    elfaces2{ii} = lns2(tris,1:3);
    elind2{ii} = unique(elfaces2{ii}(:));
    elfaces{ii} = lns2(tris,[1,3]);
end

[Node2]=MakeNode2dSmallFast(H2,g2);
[eltetra2,E2] = FindElectrodeElements2_2D(H2,Node2,elind2,2,0);
[Element2]=MakeElement2dSmallCellFast(H2,eltetra2,E2);
[H,g,Node,Element,elind,eltetra,E] = Reduce2ndOrderMesh_2D(H2,g2,elind2,2);

if plotmesh
    figure(fignum), clf, triplot(H(:,1:3),g(:,1),g(:,2)), axis image, hold on
    for ii=1:Nel
        inds = elfaces2{ii}(:);
        nds = g2(inds,:);
        %    plot(nds(:,1),nds(:,2),'o','Color','r','MarkerFaceColor','r')
        if mod(ii,2) == 1
            plot(nds(:,1),nds(:,2),'o','Color','r','MarkerFaceColor','r')
        else
            plot(nds(:,1),nds(:,2),'o','Color','m','MarkerFaceColor','m')
        end
    end
    set(gcf,'Units','normalized','OuterPosition',[0.3 0.6 0.3 0.4])
end

Mesh2.H = H2;
Mesh2.g = g2;
Mesh2.elfaces = elfaces2;
Mesh2.Node = Node2;
Mesh2.Element = Element2;

Mesh.H = H;
Mesh.g = g;
Mesh.elfaces = elfaces;
Mesh.Node = Node;
Mesh.Element = Element;

end

function [eltetra,E,nc] = FindElectrodeElements2_2D(H,Node,elnodes,order,gindx)
% [eltetra E] = FindElectrodeElements2(H,Node,elnodes,order,gindx)
%
%   elnodes{l} = indices to nodes under electrode l (cell array, size of Nel)
%   eltetra{l} = indices to elements under electrode l (cell array, size of Nel)
%   E(ii,:)    = face indices if element ii is under some electrode, zero otherwise
%   gindx      = (optional) reindexing basing on geometry, 'yes' or 'no'. 
%                  This has no effect if 1st order basis is used. Default = 'yes'
%   nc         = how many indices were changed (related to gindx)


nc = 0;

% nJ=2 (1st order elements), nJ=3 (2nd order elements)
nJ = 2;  % default
if nargin>3,
  if order==2,
    nJ = 3;
  elseif order~=1
    disp('order not supported, using default')
  end
end
if nargin<5, gindx = 'yes'; end

nH = size(H,1);

Nel = prod(size(elnodes));
if size(elnodes,2)>1
  elnodes = reshape(elnodes,Nel,1);
end


E = zeros(nH,nJ,'uint32');
eltetra = cell(Nel,1);

tetra_mask = zeros(nH,1,'uint8');

% loop through every electrode
for ii=1:Nel
  ind_node = elnodes{ii};  
  node_len = length(ind_node);
  
  % loop through nodes in electrode ii
  for jj=1:node_len
    ind_tetra = Node(ind_node(jj)).ElementConnection;    
    tetra_len = length(ind_tetra);
    
    % check every tetrahedron connected to node ptr *ii
    for kk=1:tetra_len
      tetra = ind_tetra(kk);
      Hind = H(tetra,:);
      
      if ~tetra_mask(tetra)  % ...we want select the tetrahedron only once
        [C,II] = intersect(Hind,ind_node);
        if (length(C)==nJ)
          eltetra{ii} = [eltetra{ii};uint32(tetra)];
%           E(tetra,:) = Hind(sort(II)); %works when the last three columns
%           of H are the 2nd order nodes

            E(tetra,:) = sort(Hind(II));
            if order == 2
                E(tetra,:) = E(tetra,[1,3,2]);
            end
        end      
      end
      tetra_mask(tetra)=1;
    end
    
  end
  
end

if (strcmpi(gindx,'yes')) && (order==2), [E,nc] = reindex(E,Node); end
end

function [E,nc] = reindex(E,Node)
% reindex faces basing on geometry 
% note: this may fail with severely skewed elements

gN = max(size(Node));
g = reshape([Node.Coordinate],3,gN)'; %Nodes
nE = size(E,1);

nc = 0;
for ii=1:nE
  if all(E(ii,:))
    nodes = E(ii,:);
    mp = nodes(1:3);
    cp = .5*(g(mp,:) + g(mp([2 3 1]),:));
    gg = g(nodes(4:6),:);  % center nodes (2nd order)
    [m I1] = min((cp(1,1)-gg(:,1)).^2 + (cp(1,2)-gg(:,2)).^2 + (cp(1,3)-gg(:,3)).^2);
    [m I2] = min((cp(2,1)-gg(:,1)).^2 + (cp(2,2)-gg(:,2)).^2 + (cp(2,3)-gg(:,3)).^2);
    [m I3] = min((cp(3,1)-gg(:,1)).^2 + (cp(3,2)-gg(:,2)).^2 + (cp(3,3)-gg(:,3)).^2);
    nodes2 = [mp nodes([I1 I2 I3]+3)];
    E(ii,:) = nodes2;
    if ~isequal(nodes,nodes2), nc = nc + 1; end
  end
end

disp(['Reindexing: ' num2str(nc) ' face(s) changed!'])

end

function [gnew,Hnew,trisnew] = fixIndices2nd_2D(g,H,lns)
    %This function re-indexes mesh nodes to work with EIT codes, which
    %assume the first N node indices are the main element nodes, and the 
    %2nd order nodes come after starting at index N+1.

    Inds2nd = H(:,4:end);
    Inds2nd = unique(Inds2nd(:));
    
    Inds1st = H(:,1:3);
    Inds1st = unique(Inds1st(:));
    
    gnew = [g(Inds1st,:); g(Inds2nd,:)];
    
    %fix H indices (might be slow)
    Hnew = H;
    for ii=1:size(H,1)
        for jj=1:3 %main vertices
             ind = H(ii,jj);
             [i] = find(Inds1st == ind);
             Hnew(ii,jj) = i;
        end
        for jj=4:6 %2nd order vertices
            ind = H(ii,jj);
            [i] = find(Inds2nd == ind);
            Hnew(ii,jj) = length(Inds1st) + i;
        end
    end
    
    %fix triangle indices
    trisnew = lns;
    for ii=1:size(lns,1)
        for jj=1:2 %main vertices
             ind = lns(ii,jj);
             [i] = find(Inds1st == ind);
             trisnew(ii,jj) = i;
        end
        for jj=3 %2nd order vertices
            ind = lns(ii,jj);
            [i] = find(Inds2nd == ind);
            trisnew(ii,jj) = length(Inds1st) + i;
        end
    end
end

function [Element]=MakeElement2dSmallCellFast(H,eltetra,E)

% Function [Element]=MakeElement3dsmall(H,bg,E); 
% computes the Element data for MeshData. 
% Element is a structure including the topology (H) of each element,
% faces of each element, adjacent element of each face and information
% whether the face is internal or boundary face.
% elind includes the boundary node indices that are under the electrodes
% E includes the eleement indices that are under the electrodes

% M. Vauhkonen 10.10.1998. Modified for EIDORS3D 28.1.2000 by 
% M. Vauhkonen, University of Kuopio, Finland.

% Modified to use cell arrays 6.7.2006 by K. Karhunen, University
% of Kuopio, Finland. This modification allows arbitrary number of
% elements under each electrode.

% Speed improvement, 26.07.2006 K. Karhunen

[rH,cH]=size(H);
nel = size(eltetra,1);

c = num2cell(H,2);
Element = cell2struct(c,'Topology',2);
clear c

[Element(1:rH).Electrode] = deal([]);

for k=1:nel
  id = eltetra{k};
  for n=1:length(id)
    ii = id(n);
    Element(ii).Electrode{1}=k;
    Element(ii).Electrode{2}=[E(ii,:)];
  end  
end
 
end

function [Node]=MakeNode2dSmallFast(H,g)

% Function [Node]=MakeNode3dSmall(Element,g);
% computes the Node data for MeshData.
% Node is a structure including all the nodal coordinates and
% for each node there is information to which nodes (NodeConnection) 
% and elements (ElementConnection) the node is
% connected.  

% Original: M. Vauhkonen, University of Kuopio, Finland, 11.8.1999 
% Fast version with Cell-arrays by K. Karhunen, University of Kuopio, Finland, 6.7.2006

[rg,cg]=size(g);
msE = size(H,1);
rowlen = size(H,2);

maxlen = 10; % don't change
econn = zeros(rg,maxlen+1);
econn(:,1) = 1; %this will be incremented, each node is connected to at least two elements
for k=1:msE  %loop over elements
  id = H(k,:); %node indices of this element
  idlen = econn(id,1); %current number of elements these nodes are connected to
  if max(idlen)==maxlen %more connected elements to one or more of these nodes than expected
    maxlen = maxlen + 10;
    swap = zeros(rg,maxlen+1);
    swap(:,1:maxlen-9) = econn;
    econn = swap;
  end
  econn(id,1) = idlen + 1; %icrement the connected elements count for these nodes
  for ii=1:rowlen
    econn(id(ii),idlen(ii)+1) = k; %for these nodes, store the index of this connected element
  end  
end
clear swap 

c = num2cell(g,2);
Node = cell2struct(c,'Coordinate',2);
clear c

for k=1:rg
 elen = econn(k,1); 
 Node(k).ElementConnection = [econn(k,2:elen)];
end

end

function [Node]=MakeNode3dSmallFast(H,g)

% Function [Node]=MakeNode3dSmall(Element,g);
% computes the Node data for MeshData.
% Node is a structure including all the nodal coordinates and
% for each node there is information to which nodes (NodeConnection) 
% and elements (ElementConnection) the node is
% connected.  

% Original: M. Vauhkonen, University of Kuopio, Finland, 11.8.1999 
% Fast version with Cell-arrays by K. Karhunen, University of Kuopio, Finland, 6.7.2006

[rg,cg]=size(g);
msE = size(H,1);
rowlen = size(H,2);

maxlen = 10; % don't change
econn = zeros(rg,maxlen+1);
econn(:,1) = 1;
for k=1:msE
  id = H(k,:);
  idlen = econn(id,1);
  if max(idlen)==maxlen
    maxlen = maxlen + 10;
    swap = zeros(rg,maxlen+1);
    swap(:,1:maxlen-9) = econn;
    econn = swap;
  end
  econn(id,1) = idlen + 1;
  for ii=1:rowlen
    econn(id(ii),idlen(ii)+1) = k;
  end  
end
clear swap 

c = num2cell(g,2);
Node = cell2struct(c,'Coordinate',2);
clear c

for k=1:rg
 elen = econn(k,1); 
 Node(k).ElementConnection = [econn(k,2:elen)];
end

end

function [H,g,Node,Element,elind,eltetra,E] = Reduce2ndOrderMesh_2D(H2,g2,elind2,format)
% Reduces mesh from 2nd to 1st order basis
% Netgen compatible mesh format expected (linear elements are at the top)

Nel = length(elind2);

if format == 1 %netgen compatible format, main vertices of the elements are listed in g first, 2nd order nodes after
H = H2(:,1:3);
ng = max(H(:));
g = g2(1:ng,:);
elind = cell(Nel,1);
for ii=1:Nel
  I = elind2{ii};
  J = find(I<=ng);
  elind{ii} = I(J);
end
Node = MakeNode3dSmallFast(H,g);
[eltetra,E] = FindElectrodeElements2_2D(H,Node,elind,1);
Element = MakeElement2dSmallCellFast(H,eltetra,E);

elseif format == 2 
%2nd order nodes are listed between first order nodes
     
H = H2(:,[1,3,5]);
ng = max(H(:));
g = g2(1:ng,:);
elind = cell(Nel,1);
for ii=1:Nel
  I = elind2{ii};
  J = find(I<=ng);
  elind{ii} = I(J);
end

Node = MakeNode3dSmallFast(H,g);
[eltetra,E] = FindElectrodeElements2_2D(H,Node,elind,1);
Element = MakeElement2dSmallCellFast(H,eltetra,E);

end

end


