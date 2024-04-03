clear;close all;clc


x1 = -10:0.01:10;
x2 = -10:0.01:10;

[X1, X2] = meshgrid(x1, x2);
Y_ = obj(X1(:), X2(:));
Y = reshape(Y_, size(X1));

figure()
mesh(X1, X2, Y);
title("Visualization of objective function")
set(gca, "FontSize", 18)

figure()
contour(X1, X2, Y, 50);
title("Contour of the objective function")
set(gca, "FontSize", 18)


ITER = 60;
% Why setting such learning rate(step size) should be explained.
LR = 0.1;
FOPTIMAL = 0;

gd_obj_update = zeros(ITER, 1);
gd_err_update = zeros(ITER, 1);

ngd_obj_update = zeros(ITER, 1);
ngd_err_update = zeros(ITER, 1);

init_x1 = 10;
init_x2 = -10;

gd_x1 = init_x1;
gd_x2 = init_x2;

x1 = init_x1;
x2 = init_x2;

z1 = init_x1;
z2 = init_x2;

y1 = init_x1;
y2 = init_x2;

figure("Position",[1000, 1000, 1000, 800])
ax1 = subplot(1, 2, 1);
contour(X1, X2, Y, 50);
title("Gradient Descent")
hold on 
scatter(0, 0, 100, "k", "d", "filled")
set(gca, "FontSize", 18)

ax2 = subplot(1, 2, 2);
contour(X1, X2, Y, 50);
title("Nesterov's Accelerated Gradient Descent")
hold on 
scatter(0, 0, 100, "k", "d", "filled")
set(gca, "FontSize", 18)

for i = 1: ITER
    %% Gradient descent
    gd_obj_update(i) = obj(gd_x1, gd_x2);
    gd_err_update(i) = gd_obj_update(i) - FOPTIMAL;
    [gd_update_x1, gd_update_x2] = gradient_descent(gd_x1, gd_x2, LR);
    gd_x1 = gd_update_x1;
    gd_x2 = gd_update_x2;
    
    
    scatter(ax1, gd_x1, gd_x2, 50, "s" ,"blue", "filled");
    tth = text(ax1, 0, 8, sprintf("Iter #%d", i), "FontSize", 18);
 
    
    %% Nesterov's accelerated gradient descent
    ngd_obj_update(i) = obj(y1, y2);
    ngd_err_update(i) = ngd_obj_update(i) - FOPTIMAL;
    [update_x1, update_x2, update_y1, update_y2, update_z1, update_z2] = nesterov_acceleraed_gradient_descent(x1, x2, z1, z2, LR, i);
    
    x1 = update_x1;
    x2 = update_x2;

    z1 = update_z1;
    z2 = update_z2;

    y1 = update_y1;
    y2 = update_y2;

    subplot(1, 2, 2)
    scatter(ax2, y1, y2, 50, "red", "filled");

    pause(1)
    delete(tth);
end

hold off

%figure()
%plot(1:ITER, obj_update);
figure()
plot(1:ITER, gd_err_update, ":", "LineWidth", 4);
hold
plot(1:ITER, ngd_err_update, "-.", "LineWidth", 4);

legend("Gradient Descent", "Nesterov's Accelerated Gradient Descent");
xlabel("#Iteration")
h = ylabel("Error: $f(x_t) - f(x^*)$");
set(h,'Interpreter','latex') 
set(gca, "fontsize", 18);
grid on


%% Useful functions
%%  Nesterov's accelerated gradient descent
function [update_x1, update_x2, update_y1, update_y2, update_z1, update_z2] = ...
    nesterov_acceleraed_gradient_descent(x1, x2, z1, z2, lr, iter)
    
    [grad1, grad2] = grad(x1, x2);
    update_y1 = x1 - lr * grad1;
    update_y2 = x2 - lr *grad2;

    update_z1 = z1 - lr*(iter+1)/2*grad1;
    update_z2 = z2 - lr*(iter+1)/2*grad2;

    update_x1 = (iter+1)/(iter+3)*update_y1 + 2/(iter+3)*update_z1;
    update_x2 = (iter+1)/(iter+3)*update_y2 + 2/(iter+3)*update_z2;
    
end

%% Gradient descent
function [update_x1, update_x2] = gradient_descent(x1, x2, lr)
    [grad1, grad2] = grad(x1, x2);
    update_x1 = x1 - lr*grad1;
    update_x2 = x2 - lr*grad2;
end

%% Caculate the objective function
function [y] = obj(x1, x2)
    y = 0.26.*(x1.^2+x2.^2) - 0.48.*x1.*x2;
end

%% Caculate the gradients
function [grad1, grad2] = grad(x1, x2)
    grad1 = 0.52*x1 - 0.48*x2;
    grad2 = 0.52*x2 - 0.48*x1;
end