final_df = readtable("C:/Users/as comp/Downloads/merged_entsoe_data.csv");

% Make sure 'country' is categorical
if ~iscategorical(final_df.country)
    final_df.country = categorical(final_df.country);
end

% Get list of unique countries
uniqueCountries = unique(final_df.country);

% Loop through each country
for i = 1:numel(uniqueCountries)
    country = uniqueCountries(i);
    
    % Filter rows for this country
    subset = final_df(final_df.country == country, :);
    
    % Create new figure
    figure;
    
    % Draw histogram of 'load_actual_entsoe_transparency'
    histogram(subset.load_actual_entsoe_transparency, 'Normalization', 'pdf');
    hold on;
    
    % Add KDE (smooth distribution estimate)
    try
        % fitdist can fail if there are NaNs or constant values
        pd = fitdist(subset.load_actual_entsoe_transparency, 'Kernel');
        x_vals = linspace(min(subset.load_actual_entsoe_transparency), ...
                          max(subset.load_actual_entsoe_transparency), 200);
        y_vals = pdf(pd, x_vals);
        plot(x_vals, y_vals, 'r', 'LineWidth', 1.5);
    catch
        disp(['Skipping KDE for ', char(country), ' (insufficient data)']);
    end
    
    hold off;
    title(sprintf('Load Distribution for %s', char(country)));
    xlabel('Load (Actual)');
    ylabel('Frequency');
end
