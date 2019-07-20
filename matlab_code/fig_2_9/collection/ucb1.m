classdef ucb1 <handle
    %   Detailed explanation goes here
     properties
        m, mean, N
    end
    
    methods 
        function self=init(self,m)
            self.m =m;
            self.mean=0;
            self.N=0;
        end
        function n_pull = pull(self)
            n_pull = randn(1)+self.m;
        end
        
        function update(self,y)
            self.N = self.N+1;
            self.mean=(1-1.0/self.N)*self.mean+(1.0/self.N)*y;
            
        end
    end
end

