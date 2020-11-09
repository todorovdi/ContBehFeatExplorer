function output = projectPtOnBrainSurf(hdm, dirs, alg, printLog)


  %assume dirs is array npts x 3 of points = directions from zero
  fprintf('Starting finding intersections with alg=%s\n',alg)
  dirs_out = dirs(:,:);

  if strcmp(alg,'ray')
    orig = [0 0 0];
    tris = hdm.bnd.tri;
    for i = 1:length(dirs)
      dir = dirs(i,:);
      for j = 1:length(tris)
        tri_inds = tris(j,:);
        vecs = hdm.bnd.pos(tri_inds,:);
        v1 = vecs(1,:); v2=vecs(2,:); v3=vecs(3,:);

        [isec, t,u,v,xcoor] = TriangleRayIntersection(orig, dir, v1,v2,v3, ... 
          'lineType','ray','fullReturn',1);
        if isec && t > 0
          dir = dir * t * 0.95;
          dirs_out(i,:) = dir;
        end
        if isec && printLog
          fprintf('coordInd %d / %d, numTri %d, %d, dist %f\n',i,length(dirs),j,isec,t)
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
  elseif strcmp(alg,'nearest')
    %grid_to_use.pos(1:50,:)
    FV = [];
    FV.faces    = hdm.bnd.tri;
    FV.vertices = hdm.bnd.pos;
    maxDist = 0;
    %, faces2, vertices2, corresponding_vertices_ID, new_faces_ID
    [distances,surface_points] = point2trimesh(FV, 'QueryPoints', dirs, 'MaxDistance', maxDist, 'Algorithm','parallel');
    desired_dist = 0.5;

    d = (ones(length(distances), 1) ./ distances)';
    dd = desired_dist * [ d; d; d]';
    dirs_out = surface_points +  (surface_points - dirs) .* dd  ;  % I want points 0.5 cm inside the shell
    
  else
    fprintf('Unknown alg, doing nothing\n');
    dirs_out = dirs;
  end

  output = dirs_out;
end
