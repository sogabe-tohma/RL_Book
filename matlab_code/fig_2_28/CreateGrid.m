function CreateGrid(row,col,strt,gol,barier,n_gol)

xsp = 1 / (col + 2);
ysp = 1 / (row + 2);

x = zeros(1, 2*(col + 1));
y = zeros(1, 2*(col + 1));
i = 1;
for xi = xsp:xsp:1 - xsp
    x(2*i - 1) = xi; x(2*i) = xi;
    if(mod(i , 2) == 0)
        y(2*i - 1) = ysp;y(2*i) = 1-ysp;
    else
        y(2*i - 1) = 1 - ysp;y(2*i) = ysp;
    end
    i = i + 1;
end

x2 = zeros(1, 2*(row + 1));
y2 = zeros(1, 2*(row + 1));
i = 1;
for yi = ysp:ysp:1 - ysp
    y2(2*i - 1) = yi; y2(2*i) = yi;
    if(mod(i , 2) == 0)
        x2(2*i - 1) = xsp;x2(2*i) = 1-xsp;
    else
        x2(2*i - 1) = 1 - xsp;x2(2*i) = xsp;
    end
    i = i + 1;
end

plot(x, y, '-');
hold on
plot(x2, y2, '-');
axis([0 1 0 1]);
axis off
set(gcf, 'color', 'white');

start.row = strt.row;
start.col = strt.col;
start;
goal.row= gol.row;
goal.col = gol.col;
goal;

N_goal.row= n_gol.row;
N_goal.col = n_gol.col;
N_goal;
barrier.row=barier.row;
barrier.col=barier.col;
barrier;
setStart(start.row, start.col, row,col)
setGoal(goal.row, goal.col, row, col)
setN_goal(N_goal.row, N_goal.col, row, col)
setBarrier(barrier.row,barrier.col,row,col)

function setStart(x,y, matrixRow,matrixCol)
xsp = 1 / (matrixCol + 2);
ysp = 1 / (matrixRow + 2);
xcor = ((2*y + 1) / 2) * xsp;
ycor = 1 - (((2*x + 1) / 2) * ysp);
xcor = xcor - xsp/30;
text(xcor,ycor, 'Start')


function setGoal(x,y, matrixRow,matrixCol)
xsp = 1 / (matrixCol + 2);
ysp = 1 / (matrixRow + 2);
xcor = ((2*y + 1) / 2) * xsp;
ycor = 1 - (((2*x + 1) / 2) * ysp);
xcor = xcor - xsp/5;
text(xcor,ycor, 'Goal')

function setN_goal(x,y, matrixRow,matrixCol)
xsp = 1 / (matrixCol + 2);
ysp = 1 / (matrixRow + 2);
xcor = ((2*y + 1) / 2) * xsp;
ycor = 1 - (((2*x + 1) / 2) * ysp);
xcor = xcor - xsp/5;
text(xcor,ycor, 'N_g_oal')




function setBarrier(x,y, matrixRow,matrixCol)
xsp = 1 / (matrixCol + 2);
ysp = 1 / (matrixRow + 2);
xcor = ((2*y + 1) / 2) * xsp;
ycor = 1 - (((2*x + 1) / 2) * ysp);
xcor = xcor - xsp/5;
text(xcor,ycor, 'Barrier')

