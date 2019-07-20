function pos = getNext(now, action,row,col)
    current.row=now(1);
    current.col=now(2);
    last=current;
    pos.row=now(1);
    pos.col=now(2);
    switch action
        case 3 %east
            pos.col= current.col + 1;
        case 2 %south
            pos.row= current.row + 1;
        case 4 %west
            pos.col= current.col - 1;
        case 1 %north
            pos.row= current.row - 1;
    end
    if (pos.col==3 && pos.row==2)
        pos.row=last.row;
        pos.col=last.col;
    end
    if(pos.col <= 0)
        pos.col = 1; end
    if(pos.col > col)
        pos.col = col; end
    if(pos.row <= 0)
        pos.row = 1; end
    if(pos.row > row)
        pos.row = row; end
    pos=[pos.row,pos.col];
end