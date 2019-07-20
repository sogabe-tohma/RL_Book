function ParseArrows(arr, matrixRow, matrixCol,start,goal,barrier,N_goal)

CreateGrid(matrixRow, matrixCol,start,goal,barrier,N_goal)
for row = 1:matrixRow
    for col = 1:matrixCol
        for act = 1:4
            if (arr(row,col,act) == 1)
                DrawArrow(act, row, col, matrixRow, matrixCol)
            end
        end
    end
end

function DrawArrow(act, row, col, matrixRow, matrixCol)
rotation = 0;
textToDraw = 'o';

switch act
   case 1 % east
       textToDraw = '\uparrow';
       rotation = 0;
   case 2 % south
       textToDraw = '\downarrow';
       rotation = 0;
   case 3 % west
       textToDraw = '\leftarrow';
       rotation = 0;
   case 4 % north
       textToDraw = '\rightarrow';
       rotation = 0;
   case 5 % hold
       textToDraw = 'o';
       rotation = 0;
   otherwise
      disp(sprintf('invalid action index: %d', act))
end

xsp = 1 / (matrixCol + 2);
ysp = 1 / (matrixRow + 2);
xcor = ((2*col + 1) / 2) * xsp;
ycor = 1 - (((2*row + 1) / 2) * ysp);
xcor = xcor - xsp/5;
text(xcor, ycor, textToDraw, 'Rotation', rotation)
