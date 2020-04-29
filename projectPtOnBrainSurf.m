function output = projectPtOnBrainSurf(hdm, dirs, printLog)
  %assume dirs is array npts x 3 of points = directions from zero
  dirs_out = dirs(:,:);

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
        fprintf('coordInd %d, numTri %d, %d, dist %f\n',i,j,isec,t)
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

  output = dirs_out;
end
