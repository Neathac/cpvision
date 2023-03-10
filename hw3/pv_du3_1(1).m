%% 2D Kalman filter

% Naprogramujte sledovanie auta vo videu. Mozete zacat s car1.mp4, kde je
% zelene pozadie a detekcia auta je jednoduchsia

% nastavenie videovstupu
vidReader = VideoReader('car1.mp4','CurrentTime',0);
oldFrame=im2double(rgb2gray(readFrame(vidReader)));
[m,n,~]=size(oldFrame);


% inicializujte premenne Kalmanovho filtra

% stav auta (x,y,rychlost x, rychlost y)
% zrychlenie v smere x a y je 2D nahodna premenna s kovariancnou maticou S
% potom Q=G*S*G';
% mozete pouzit S=[1 0;0 1];
% matice A a G vytvorte podla 1D prikladu z prednasky
% mozete pouzit R=[1 0;0 1];

% skutocny stav auta nepozname, vieme len zistit poziciu - v kazdom frame
% treba auto najst

% ak chcete vediet viac, ako sa odhaduju kovariancne matice sumu S a R
% https://onlinelibrary.wiley.com/doi/full/10.1002/acs.2783
% nie je sucastou domacej ulohy

% prejdite cele video
while hasFrame(vidReader)
    % nacitajte frame
    frameGray = im2double(rgb2gray(readFrame(vidReader)));
    
    % detekujte auto
    % vykreslite nameranu polohu
    % spocitajte Kalmana
    % vykreslite odhadnutu a korigovanu polohu

end