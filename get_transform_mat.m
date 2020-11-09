function M = get_transform_mat(subjstr)
  data_dir = getenv('DATA_DUSS');
  basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
  fname_head = strcat(data_dir, basename_head );
  hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

  srsstd = load('~/soft/fieldtrip/template/sourcemodel/standard_sourcemodel3d5mm'); % 5mm spacing
  vecinds = [8157       16509          99       22893];          % don't touch these indices!
  preX = srsstd.sourcemodel.pos;   % in each ROW -- x,y,z
  preY = hdmf.mni_aligned_grid.pos;
  % M * X = Y 
  X0 = preX( vecinds, : );  %3x3
  Y0 = preY( vecinds, : );  %3x3

  X1 = transpose(X0);
  Y1 = transpose(Y0);

  %X = [X1; [0 0 0 1] ];
  %Y = [Y1; [0 0 0 1] ];
  X = [X1; [1 1 1 1] ];
  Y = [Y1; [1 1 1 1] ];

  d = det(Y);
  if abs(d) < 1e-10
    printf("Bad selection of vectors")
    return
  end

  M = Y * inv(X);
  %M = Y \ X
  tryinds = [7 13 567];
  ii = 2;
  dev = M * transpose( [ preX(tryinds(ii), :) 1] ) - transpose( [preY(tryinds(ii),:) 1] );
  devv = dev(1:3);
  fprintf('dev = %d\n',devv)
  %%%%%%%%%%%% end coord sys conv
end
