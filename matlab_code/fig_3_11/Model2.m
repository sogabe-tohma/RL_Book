classdef Model2 <handle
    %actions=["up";"down";"right";"left"]
    % grid is initialize with grid initialization
    properties(Access = public)
        theta=randn(1,25)/sqrt(25);
    end
    methods
        function SS=sa2x(self,s,a)
            ss=[];
           if a==1 ss=[ss,(s(2)-2)];else ss=[ss,0];end  
           if a==1 ss=[ss,(s(1)-2.5)]; else ss=[ss,0]; end 
           if a==1 ss=[ss,((s(1)*s(2)-4)/4)]; else ss=[ss,0]; end 
           if a==1 ss=[ss,((s(2)*s(2)-3)/3)];else ss=[ss,0];end
           if a==1 ss=[ss,((s(1)*s(1)-5.5)/5.5)];else ss=[ss,0];end
           if a==1 ss=[ss,1];else ss=[ss,0];end
           if a==2 ss=[ss,(s(2)-2)];else ss=[ss,0];end  
           if a==2 ss=[ss,(s(1)-2.5)]; else ss=[ss,0]; end 
           if a==2 ss=[ss,((s(1)*s(2)-4)/4)]; else ss=[ss,0]; end 
           if a==2 ss=[ss,((s(2)*s(2)-3)/3)];else ss=[ss,0];end
           if a==2 ss=[ss,((s(1)*s(1)-5.5)/5.5)];else ss=[ss,0];end
           if a==2 ss=[ss,1];else ss=[ss,0];end
           if a==3 ss=[ss,(s(2)-2)];else ss=[ss,0];end  
           if a==3 ss=[ss,(s(1)-2.5)]; else ss=[ss,0]; end 
           if a==3 ss=[ss,((s(1)*s(2)-4)/4)]; else ss=[ss,0]; end 
           if a==3 ss=[ss,((s(2)*s(2)-3)/3)];else ss=[ss,0];end
           if a==3 ss=[ss,((s(1)*s(1)-5.5)/5.5)];else ss=[ss,0];end
           if a==3 ss=[ss,1];else ss=[ss,0];end
           if a==4 ss=[ss,(s(2)-2)];else ss=[ss,0];end  
           if a==4 ss=[ss,(s(1)-2.5)]; else ss=[ss,0]; end 
           if a==4 ss=[ss,((s(1)*s(2)-4)/4)]; else ss=[ss,0]; end 
           if a==4 ss=[ss,((s(2)*s(2)-3)/3)];else ss=[ss,0];end
           if a==4 ss=[ss,((s(1)*s(1)-5.5)/5.5)];else ss=[ss,0];end
           if a==4 ss=[ss,1];else ss=[ss,0];end
           ss=[ss,1];
           SS=ss;
        end
        function cur=grad(self,s,a)           
             cur=self.sa2x(s,a);
        end
        
        function pos=predict(self,state,action)
            x=self.sa2x(state,action);
            pos=dot(self.theta,x);
        end
       
    end
    
end

