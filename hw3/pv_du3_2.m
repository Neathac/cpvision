%% priprava 1

% stiahnite si databazu obrazov (111MB)
% http://imagedatabase.cs.washington.edu/groundtruth/icpr2004.imgset.rar
% budete pracovat s adresarom \icpr2004.imgset\groundtruth


% vytvorenie paletety farieb
N=4;
delta = 1/N;
vecR = (0:delta:1-delta)+delta/2;
vecG = vecR;
N=3;
delta = 1/N;
vecB = (0:delta:1-delta)+delta/2;
[R,G,B] = meshgrid(vecR,vecG,vecB);
% tuto paletu budete pouzivat v nasledovnych castiach
paleta = [R(:) G(:) B(:)];
% pocet farieb v palete
hist_len = size(paleta,1);

%% priprava 2

% vytvorte imageDataStore objekt z ulozenej databazy
% obrazky su v subfolderoch
path_to_images = '';
imagePath = [path_to_images, '\icpr2004.imgset\groundtruth'];
imds = imageDatastore(imagePath,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');


%% predspracovanie

% konstanta pre gridded_color
K=4;
% predspracujte obrazky a spocitajte rozne priznaky
NrIm = size(imds.Files,1);
% do pola priznaky budeme zapisovat histogram farieb dlzky hist_len a
% priemernu farbu v KxK gridoch
priznaky = zeros(NrIm,hist_len+K*K*3);

for ii = 1:NrIm
    img = readimage(imds,ii);
    img = im2double(img);
    % obrazok prevedte na indexovany pomocou palety, nezabudnite pouzit 'nodither'
    img_ind = 0;

    % vyrobte normalizovany histogram farieb (histcounts), suma_prvkov=1
    color_hist = 0;
    
    % ulozte histogram
    priznaky(ii,1:hist_len)=color_hist;
    
    % rozdelte obrazok na mriezku 4x4

    for jj=1:K
        for kk=1:K
            % pre kazdy podobrazok vypocitajte priemernu farbu (z povodnych)

            % ulozte 
            priznaky(ii,hist_len+1:end) = 0;
        end
    end
    
    
end

%% uloha  

% urcite si index query obrazku Q
% najdite 9 najpodobnejsich obrazkov ku query obrazku na zaklade podmienok a) b) a c)
% V casti a) a b) identifikujte 3 najcastejsie farby palety query obrazku. Toto budu farby, ktore nas zaujimaju - predstavte si, ze histogram ma len tieto 3 stlpce. Pracujte len s tymto orezanym histogramom.
% a) Len pre tieto tri farby spocitajte sumu euklidovskej vzdialenosti histogramov 
% b) Len pre tieto tri farby spocitajte prienik histogramov 
% c) Rozdelte obrazok na mriezku. Pouzite vhodnu velkost podla velkosti obrazkov. Vyberte vhodneho reprezentanta jednotlivych buniek mriezky podla prednasky. Spocitajte gridded_color vzdialenost 

% Mozete puzit funkcie mink a maxk - podla definicie podobnosti pre danu
% metriku
% Vykreslite pomocou prikazu subplot

Q=68;
query_img = readimage(imds,Q);
query_priznaky = priznaky(Q,:);



