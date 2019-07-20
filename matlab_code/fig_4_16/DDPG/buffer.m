classdef buffer < handle   
    properties (SetAccess=public)
        inter=0;
        count=1;
        memo=zeros(20000,7);
        memor=zeros(1,7);
        point=zeros(1,7);
    end
    methods
        function append(self,x)
            self.point=x;
            if length(self.memo)>=self.count
                for in=1:length(self.point)
                    self.memor(self.count,in)= self.point(in);
                end
                self.count= self.count+1;
            else
                self.memo(1:(length(self.memo))-1,:)= self.memor(2:length(self.memo),:);
                self.memor=self.memo;
                self.count= self.count-1;
                for in=1:length(self.point)
                    self.memor(self.count,in)= self.point(in);
                end
            end
        end
        function clear(self)
            self.memor=zeros(1,length(self.point));
            self.count=1;
        end
        function n=memory(self)
            n= self.memor;
        end
        function m = len(self)
            m=self.count-1;
        end
        function rbuf=randslc(self,s)
             WW= self.memor(randperm(size(self.memor,1)),':');
            rbuf=WW(1:s,:);
        end
    end
end



