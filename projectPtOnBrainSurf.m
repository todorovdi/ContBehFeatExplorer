function output = projectPtOnBrainSurf(hdm, dirs, alg, desired_dist,project_only_outside_points, printLog)

  if ~exist('printLog','var')
    printLog=1;
  end

  if ~exist('desired_dist')
    desired_dist = 0.5;
  end

  if ~exist('project_only_outside_points')
    project_only_outside_points = 0;
  end
  %assume dirs is array npts x 3 of points = directions from zero
  fprintf('Starting finding intersections with alg=%s\n',alg)
  dirs_out = dirs(:,:);

  if strcmp(alg,'ray')
    nprojected = 0;

    orig = [0 0 0];
    tris = hdm.bnd.tri;
    % cycle over vectors to intersect with the triangles
    for i = 1:length(dirs)
      dir = dirs(i,:);
      veclen_orig = norm(dir);
      % cycle over triangles
      for j = 1:length(tris)
        tri_inds = tris(j,:);
        vecs = hdm.bnd.pos(tri_inds,:);
        v1 = vecs(1,:); v2=vecs(2,:); v3=vecs(3,:);

        % t -- distance
        [isec, t,u,v,xcoor] = TriangleRayIntersection(orig, dir, v1,v2,v3, ... 
          'lineType','ray','fullReturn',1);
        if isec && t > 0
          if t < 1 ||  ~project_only_outside_points
            % t * dir -- intersection point
            % I want len(t * dir - newpt) = desired_dist
            %shift = sqrt(desired_dist^2 / 3) / veclen_orig
            %coef = 0.95;
            %coef = 1 - shift
            coef = 1 - desired_dist / veclen_orig;
            dir = dir * t * coef;
            %dir = dir * t * 0.95;  % forcing to be inside, a bit below the trianlge
            dirs_out(i,:) = dir;
            nprojected = nprojected + 1;
          end
        end
        if isec && printLog
          fprintf('projectPtOnBrainSurf: coordInd %d of %d, numTri=%d, isec=%d, dist=%f\n',i,length(dirs),j,isec,t)
        end


        %[isec, t,u,v,xcoor] = TriangleRayIntersection(orig, dir, v1,v2,v3, ... 
        %  'lineType','segment','fullReturn',1);
        %if isec & t > 0.8 
        %  sprintf('0_coordInd %d, numTri %d, %d, dist %f',i,j,isec,t)
        %  dir = dir * t * 0.95;
        %  dirs_out(i,:) = dir;
        %  [isec, t,u,v,xcoor] = TriangleRayIntersection(orig, dir, v1,v2,v3, ... 
        %    'lineType','segment','fullReturn',1);
        %  if isec && printLog
        %    sprintf('coordInd %d, numTri %d, %d, dist %f',i,j,isec,t)
        %  end
        %end
      end
    end
    fprintf('projectPtOnBrainSurf: Projected %d points ', nprojected );
  elseif strcmp(alg,'nearest')
    %grid_to_use.pos(1:50,:)
    FV = [];
    FV.faces    = hdm.bnd.tri;
    FV.vertices = hdm.bnd.pos;
    maxDist = 0;
    %, faces2, vertices2, corresponding_vertices_ID, new_faces_ID
    [distances,surface_points] = point2trimesh(FV, 'QueryPoints', dirs, 'MaxDistance', maxDist, 'Algorithm','parallel');

    % distances can be negative! (it depends on surface normal). Negative means that the orig point is inside
    % where surface points lengths (as vectors from zero) are shorter than dirs
    if boolean( project_only_outside_points)
      mask = distances' > 0;
    else
      mask = true( length(distances) , 1 );
    end

    dirs_out = dirs;
    % 1/distances
    d = (ones(length(distances), 1) ./ distances)';
    dd = desired_dist * [ d; d; d]';
    %size(dd)
    %size(surface_points)
    %size(dirs)
    %(surface_points(mask,:) - dirs(mask,:) ) .* dd(mask,:)
    dirs_out(mask,:) = surface_points(mask,:) +  (surface_points(mask,:) - dirs(mask,:) ) .* dd(mask,:)  ;  % I want points 0.5 cm inside the shell
    fprintf('projectPtOnBrainSurf: Projected %d points ', sum(mask) );
    
  elseif strcmp(alg,'no')
    fprintf('projectPtOnBrainSurf: alg=no, doing nothing\n');
    dirs_out = dirs;
  else
    error('projectPtOnBrainSurf: Unknown alg, exiting\n');
    dirs_out = [];
  end

  output = dirs_out;
end
