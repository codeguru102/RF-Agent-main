% Template worker for RF-Agent-MATLAB pending candidate folders.
% Copy this file into your MATLAB project and replace train_one_candidate
% with your real training/evaluation function.

function run_pending_candidates_template()
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    taskName = "matlab_task";
    taskDir = fullfile(projectRoot, "tasks", taskName);
    candidatesDir = fullfile(taskDir, "candidates");

    candidates = dir(fullfile(candidatesDir, "candidate_*"));
    for i = 1:numel(candidates)
        candidateDir = fullfile(candidates(i).folder, candidates(i).name);
        statusPath = fullfile(candidateDir, "status.json");
        if ~isfile(statusPath)
            continue;
        end

        statusText = fileread(statusPath);
        status = jsondecode(statusText);
        if ~strcmp(status.status, "pending")
            continue;
        end

        try
            mark_status(candidateDir, "running", "");
            rewardPath = fullfile(candidateDir, "reward_fcn.m");

            % Replace this placeholder with your actual MATLAB RL training.
            result = train_one_candidate(rewardPath, candidateDir);

            summaryPath = fullfile(candidateDir, "summary.json");
            write_json(summaryPath, result);
            mark_status(candidateDir, "trained", "");
        catch ME
            result = struct();
            result.status = "failed";
            result.error_message = ME.message;
            write_json(fullfile(candidateDir, "summary.json"), result);
            mark_status(candidateDir, "failed", ME.message);
        end
    end

    dashboardScript = fullfile(projectRoot, "src", "main.py");
    command = sprintf('python "%s" --mode inspect --task-dir "%s"', dashboardScript, taskDir);
    system(command);
end

function result = train_one_candidate(rewardPath, candidateDir)
    %#ok<INUSD>
    % TODO: call your real MATLAB trainer here.
    % Your trainer should use rewardPath, write logs/train.csv if available,
    % and return the metrics configured in tasks/<task_name>/task.json.
    result = struct();
    result.status = "trained";
    result.max_task_score = 0.0;
    result.final_task_score = 0.0;
    result.mean_return = 0.0;
    result.success_rate = 0.0;
    result.constraint_violation = 0.0;
    result.notes = "Placeholder result from template worker.";
end

function mark_status(candidateDir, state, errorMessage)
    status = struct();
    status.status = state;
    status.updated_at = string(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    status.error_message = errorMessage;
    write_json(fullfile(candidateDir, "status.json"), status);
end

function write_json(path, data)
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(data, PrettyPrint=true));
end
