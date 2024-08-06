
load('AllSS')
load('chialvo2013')
cmap = hsv(length(SS));
leg = {'k/(\rho_p d^3 \gamma^2) = 10^3','k/(\rho_p d^3 \gamma^2) = 10^4',...
    'k/(\rho_p d^3 \gamma^2) = 10^5','k/(\rho_p d^3 \gamma^2) = 10^6','k/(\rho_p d^3 \gamma^2) = 10^7'};
nuc = 0.634;

%--------------------------------------------------------------------------
%%%%-----------------------------------------------------------------------
% Midi 2004 - RIGID 
% I = gp*d*sqrt(rhop/p)
% mu(I) = mus + (mu2 - mus)/(I0/I + 1)
mu0 = 0.12;
muinf = 0.55;
I0 = 0.2;
nu_min = 0.44;
mu_midi = @(I) mu0 + (muinf - mu0)./(I0./I + 1);
nu_midi = @(I) nuc - (nuc - nu_min).*I;

%%%%-----------------------------------------------------------------------
% Luding
mu0 = 0.12;%0.15;
muinf = 0.55;%0.42;
I0 = 0.2;%0.06;

p0 = 0.9;
I0nu = 3.28;%0.85;
p0nu = 0.33;

mu_l = @(I,pstar) (mu0 + (muinf-mu0)./(I0./I + 1)).*(1-(pstar./p0).^(1/2));
nu_l = @(I,pstar) nuc.*(1-I./I0nu).*(1+pstar./p0nu);

p_l = logspace(-1,6,1000);
I_l = 1./sqrt(p_l);

%--------------------------------------------------------------------------
% p* = p*d/k vs nu

figure
for i=1:5
    h(i) = semilogy(SS(i).nu,SS(i).p./SS(i).k,'o','Color',cmap(i,:));
    hold all
    semilogy(nu_l(I_l,p_l./SS(i).k),p_l./SS(i).k,'Color',cmap(i,:));
end
hc = semilogy(c2013_mu0(1).nu,c2013_mu0(1).p./10^8,'k*');
semilogy(nu_l(I_l,p_l./10^8),p_l./10^8,'k');
xlim([0.5,0.7])
ylim([10^-8,10^-1])
xlabel('\nu')
ylabel('p^* = p*d/k')
legend([h,hc],[leg,'k/(\rho_p d^3 \gamma^2) = 10^8'])


%--------------------------------------------------------------------------
% s/p vs nu

figure
for i=1:5
    h(i) = plot(SS(i).nu,SS(i).s./SS(i).p,'o','Color',cmap(i,:));
    hold all
    plot(nu_l(I_l,p_l./SS(i).k),mu_l(I_l,p_l./SS(i).k),'Color',cmap(i,:));
end
hc = plot(c2013_mu0(1).nu,c2013_mu0(1).s./c2013_mu0(1).p,'k*');
semilogy(nu_l(I_l,p_l./10^8),mu_l(I_l,p_l./10^8),'k');
xlim([0.5,0.7])
ylim([0.08,0.5])
xlabel('\nu')
ylabel('s/p')
legend([h,hc],[leg,'k/(\rho_p d^3 \gamma^2) = 10^8'])

%--------------------------------------------------------------------------
% s/p vs I

figure
for i=1:5
    h(i) = semilogx(SS(i).I,SS(i).s./SS(i).p,'o','Color',cmap(i,:));
    hold all
    semilogx(I_l,mu_l(I_l,p_l./SS(i).k),'Color',cmap(i,:));
end
hc = semilogx(1./sqrt(c2013_mu0(1).p),c2013_mu0(1).s./c2013_mu0(1).p,'k*');
semilogx(I_l,mu_midi(I_l),'k') % MIDI expression
ylim([0.0,0.5])
xlim([0.001,3])
xlabel('I')
ylabel('s/p')
legend([h,hc],[leg,'k/(\rho_p d^3 \gamma^2) = 10^8'])

%--------------------------------------------------------------------------
% nu vs I

figure
for i=1:5
    h(i) = semilogx(SS(i).I,SS(i).nu,'o','Color',cmap(i,:));
    hold all
    semilogx(I_l,nu_l(I_l,p_l./SS(i).k),'Color',cmap(i,:));
end
hc = semilogx(1./sqrt(c2013_mu0(1).p),c2013_mu0(1).nu,'k*');
semilogx(I_l,nu_midi(I_l),'k') % MIDI expression
ylim([0.0,0.7])
xlim([0.001,3])
xlabel('I')
ylabel('\nu')
legend([h,hc],[leg,'k/(\rho_p d^3 \gamma^2) = 10^8'])

%--------------------------------------------------------------------------
% Fluidity/sqrt(T) : g*d/T^(1/2)
% where g = gammadot/(s/p)  fluidity

figure
for i=1:length(SS)
    h(i) = plot(SS(i).nu,(SS(i).p./SS(i).s)./SS(i).T.^(1/2),'o','Color',cmap(i,:));
    hold all
end
hc = plot(c2013_mu0(1).nu,c2013_mu0(1).p./c2013_mu0(1).s./c2013_mu0(1).T.^(1/2),'k*');

xlabel('\nu')
ylabel('g*d/(T^{1/2} = p*d*\gamma / (s*T^{1/2})')
xlim([0.,0.7])
legend([h,hc],[leg,'k/(\rho_p d^3 \gamma^2) = 10^8'])



