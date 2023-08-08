function [smagNote, smagMusic, sphaseMusic] = load_data()
%% Argument Descriptions
% Required Input Arguments:
% None

% Required Output Arguments:
% smagNote: 1025 x 11 matrix containing the mean spectrum magnitudes of the notes. A correct sequence of the notes is REQUIRED. (From left to right: e f g a b c d e2 f2 g2 a2)
% smagMusic: 1025 x K matrix containing the spectrum magnitueds of the music after STFT.
% sphaseMusic: 1025 x K matrix containing the spectrum phases of the music after STFT.

%% Load Spectrum Magnitudes of Notes
% Fill your code here to return 'smagNote'
notesfolder = './data/notes_15/';
listname = dir([notesfolder '*.wav']);
notes = [];
for k =1:length(listname)
    [s,fs] = audioread([notesfolder listname(k).name]);
    s = resample(s,16000,fs);  
    s =  s(:,1);
    spectrum = stft(s',2048,256,0,hann(2048));
    % Find the central frame
    middle = ceil(size(spectrum,2)/2);
    note = abs(spectrum(:,middle));
    % Clean up everything more than 40db below the peak
    note(find(note < max(note(:))/100)) = 0;
    note = note/norm(note); %normalize the note to unit length
    notes = [notes,note];
end
smagNote = notes;
%% Load Spectrum Magnitues and Phases of The Provided Music
% Fill your code here to return 'smagMusic' and 'sphaseMusic'
notesfolder = './data/';
listname = dir([notesfolder '*.wav']);
musics = [];
[s,fs] = audioread([notesfolder listname(1).name]);
s = resample(s,16000,fs);
s = s(:,1);
[spectrum,sphaseMusic] = stft(s',2048,256,0,hann(2048));
music   = abs(spectrum);
% Clean up everything more than 40db below the peak
music(find(music < max(music(:))/100)) = 0;
music = music/norm(music); %normalize the note to unit length
musics = [musics,music];
smagMusic = musics;

end