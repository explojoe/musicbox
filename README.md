# Music Box
This project is centered around fft.py, a program that analyzes audio for piano tones and then outputs a piano tone locked version of the input.

## Running

```
python fft.py <input_file.wav> <analysis_chunk_number>
```
Will process `input_file.wav` and create/overwrite `output.wav`. `analysis_chunk_number` can be passed to specify a particular chunk to analyze the frequency spectrum of. 

### Example
Here is an example output of the program.

![Example Output](https://i.imgur.com/cT2A95M.png)

From top to bottom there are 6 charts:
1. Input waveform.
* Thick blue line indicates middle of target chunk.
2. Waveform of target chunk window (chunk + `WINDOW_SIZE` on each side).
3. Frequency spectrum of target chunk window.
* Vertical thick red line indicates chosen frequency after Harmonic Sum Spectrum (HSS) and parabolic peak estimation (PPE).
* Thin red line indicates the frequency range at which this program is optimised for.
* Thin black lines indicate octave boundaries.
4. HSS of target chunk window focussed on target frequency range.
* Thin black lines indicate octave boundaries.
* Dotted green lines indicate detected peaks.
* Detected peaks are recalculated using PPE.
* If the greatest peak after PPE is lower frequency than the greatest peak before PPE, keep the new peak frequency.
5. Detected frequency over time before median filter.
* Thin red line indicates the frequency range at which this program is optimised for.
* Thin black lines indicate octave boundaries.
6. Median filtered frequency over time.
* Thin red line indicates the frequency range at which this program is optimised for.
* Thin black lines indicate octave boundaries.
7. Output waveform.
* Thick blue line indicates middle of target chunk.
  

## Built With

* [SciPy](https://www.scipy.org/) - Used for filtering and window functions
* [NumPy](https://numpy.org/) - Used for arrays and data management
* [wavio](https://pypi.org/project/wavio/) - Used to convert between `.wav` and numpy arrays
* [Matplotlib](https://matplotlib.org/) - Used to generate those pretty graphs


## Authors

**Joshua Wentzel** - [explojoe](https://github.com/explojoe)
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you [Dr. Nguyen](https://eecs.oregonstate.edu/people/nguyen-thinh) for getting me interested in signals processing.
