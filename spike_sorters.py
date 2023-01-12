import spikeinterface.sorters as ss
import spikeinterface.extractors as se

ss.CombinatoSorter.set_combinato_path('C:\WORK\Sorters\combinato')
ss.Kilosort2Sorter.set_kilosort2_path('C:\WORK\Sorters\Kilosort2')

print('Available sorters', ss.available_sorters())
print('Installed sorters', ss.installed_sorters())

test_recording, _ = se.toy_example(
    duration=30,
    seed=0,
    num_channels=64,
    num_segments=1
)
print(test_recording)

# test_recording = test_recording.save(folder="test-docker-folder")
#
# sorting = ss.run_herdingspikes(recording=test_recording, detect_threshold=4)
#
# print(sorting)

