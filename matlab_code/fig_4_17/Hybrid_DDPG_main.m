sprev=rng(11111,'v5uniform');
actor=actor();
critic=critic();
doVid = false;
if doVid
    writerObj = VideoWriter('qlearnVid.mp4','MPEG-4');
    writerObj.FrameRate = 60;
    open(writerObj);
end
panel = figure;
panel.Position = [300 200 300 200];
panel.Color = [1 1 1];
subplot(1,4,1)
hold on
f = plot(0,0,'b','LineWidth',10);
axPend = f.Parent;
axPend.Visible='off';
axPend.Position = [0.2 0.2 0.3 0.3]; 
axPend.Clipping = 'off';
axis equal
axis([-1.2679 1.2679 -1 1]);
plot(0.001,0,'.k','MarkerSize',50); 
hold off
collection=0;
learning=1;
b_size=64;
tau=0.001;
global max_it;
episod=200;
max_it=1000;
r_plot=zeros(episod,1);
loss=0;
max_time=5;
counting=0;
for times=1:max_time
    buf=buffer;
    for se =1:episod
        s_t=[pi,0];
        rewrd=0;
        rewd=0;
        for je=1:max_it
            state=s_t;
            a_t = actor.predict(state,1);
            at=a_t +1/(1+se+je);
            [N_s,TERMINATE]=environment(state,at,je);
            trajectory =[state,at,N_s,TERMINATE,rewd];       
            buf.append(trajectory)
            if je>0
                set(f,'XData',[0 -sin(state(1))]);
                set(f,'YData',[0 cos(state(1))]);            
                drawnow;
            end
            rewd = (-(abs(N_s(1))).^2 + -0.25*(abs(N_s(2))).^2)/5;
            rewrd=rewrd+rewd;
            tie=buf.len;
            loss=critic.train(state,at,1);
            if tie>65
                S=buf.randslc(b_size); 
                reward=S(:,7);
                next_state=S(:,4:5);
                done=S(:,6);
                stte=S(:,1:2);                
                acton=S(:,3);
                a_tgt=actor.predict(next_state,0);
                q_tgt=critic.predict(next_state,a_tgt,0);
                y=zeros(length(S),1);
                for ie=1:b_size
                    if done(ie)
                        y(ie)=reward(ie);
                    else
                        y(ie)=reward(ie)+0.99*q_tgt(ie);
                    end
                end
                loss=loss+critic.train(stte,acton,y);
                a_for_dq_da= actor.predict(stte,1);
                if counting==0
                    dq_Da=critic.evaluate_action_gradient(stte,a_for_dq_da,1);
                    actor.train(stte,dq_Da,1);
                    counting=1;
                else
                    dl_Da=critic.evaluate_action_loss(stte,a_for_dq_da,a_tgt,1);
                    actor.train(stte,dl_Da,1);
                    counting=0;
                end
                actor.train_target(tau);
                critic.train_target(tau);                            
             end
            s_t=N_s;
        end
     fprintf('step  %d,   reward  %d\n',se,rewrd)
    end
 end

function [next,ter]=environment(sTate,T,STEP)
global  max_it
if STEP==max_it
    ter=1;
else
    ter=0;            
end
if T<-2
    T=0.8*-2;
elseif T>2
    T=0.8*2;
else 
    T=T;
end
zz1=sTate;
if norm(zz1)>=2.2
    dt=0.15;
elseif norm(zz1)<=2.19 && norm(zz1)>=1.00
    dt=0.09;
elseif norm(zz1)<=1.01 && norm(zz1)>=0.5
    dt=0.05;
elseif norm(zz1)<=0.49 && norm(zz1)>=0.1
    dt=0.02;
else
    dt=0.01;
end

for q = 1:2
    k1 = Dynamics(zz1,T);
    k2 = Dynamics(zz1+dt/2*k1,T);
    k3 = Dynamics(zz1+dt/2*k2,T);
    k4 = Dynamics(zz1+dt*k3,T);            
    z2 = zz1+ dt/6*(k1 + 2*k2 + 2*k3 + k4);
    % All states wrapped to 2pi    
    if z2(1)>pi
        z2(1) = -pi + (z2(1)-pi);
    elseif z2(1)<-pi
        z2(1) = pi - (-pi - z2(1));
    end
    if z2(2)<-8
        z2(2)=dt*z2(2);
    elseif z2(2)>8
        z2(2)=-dt*z2(2);
    else
        z2(2)=z2(2);
    end
end 
next=z2;     
end
function zdot = Dynamics(Z,T)
g = 1;
L = 1;
Z = Z';
zdot = [Z(2) g/L*sin(Z(1))+T];
end
