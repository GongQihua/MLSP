clc; 
clear all; 
close all;
%% Load Notes and Music
% Use the 'load_data' function here
[smagNote,smagMusic,sphaseMusic] = load_data();
%% Solution for Problem 2a-I here
% Your solution will give you one “score” W_i per note. Place all the 15 scores W_i (for the 15 notes) into a single matrix W. 
% Place the score for the i-th note in the i-th row of W.
% W will be a 15xT matrix, where T is the number of frames in the music.
% Store W in a file called "problem2a.dat" in the "results" folder
for i=1:size(smagNote,2)
    W(i,:) = pinv(smagNote(:,i)')'* smagMusic;
end
save('./results/problem2a.mat','W');

%% Solution to Problem 2a-II here: Synthesize Music
% Use the 'synthesize_music' function here.
% Use 'wavwrite' function to write the synthesized music as 'problem2a_synthesis.wav' to the 'results' folder.
synthesizemusic = synthesize_music(sphaseMusic,smagNote*W)';
audiowrite('./results/problem2a_synthesis.wav',20000*synthesizemusic,16000);
%The reconstituted music is much less clear and has more noise than the
%original music

%% Solution for Problem 2b-I here
% Your solution will give you singe “score” matrix W. Store W in a file called "problem2b.dat" in the 'results' folder
W15 = pinv(smagNote)* smagMusic;
save('./results/problem2b.mat','W15');

%% Solution to Problem 2b-II here:  Synthesize Music
% Use the 'synthesize_music' function here.
% Use 'wavwrite' function to write the synthesized music as 'problem2b_synthesis.wav' to the 'results' folder.
synmusic = synthesize_music(sphaseMusic,smagNote * W15)';
audiowrite('./results/problem2b_synthesis.wav',20000 * synmusic,16000);
%Pretty much the same, but it sounds like there's less musical noise
%reconstructed here.