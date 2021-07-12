task_seq_to_task_list = {
  'MT10': ['reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1',
           'drawer-open-v1', 'drawer-close-v1', 'button-press-topdown-v1',
           'peg-insert-side-v1', 'window-open-v1', 'window-close-v1'],

  'EASY4_V0': ['reach-v1', 'door-open-v1', 'drawer-close-v1',
               'button-press-topdown-v1'],

  # tasks that are easy with new codebase
  'NEW_EASY5_V0': ['reach-v1', 'drawer-close-v1', 'button-press-topdown-v1',
                   'window-open-v1', 'push-v1'],

  'NEW_EASY5_REV_V0': ['push-v1', 'window-open-v1', 'button-press-topdown-v1',
                       'drawer-close-v1', 'reach-v1'],

  'MICHAL10_V0': ['reach-v1', 'sweep-into-v1', 'push-v1', 'button-press-v1',
                  'soccer-v1', 'shelf-place-v1', 'door-open-v1',
                  'peg-insert-side-v1', 'pick-place-v1', 'pick-place-wall-v1'],

  'HARDEST10_V0': ['pick-place-v1', 'peg-insert-side-v1', 'stick-pull-v1',
                   'pick-place-wall-v1', 'peg-unplug-side-v1',
                   'disassemble-v1', 'pick-out-of-hole-v1', 'assembly-v1',
                   'push-back-v1', 'lever-pull-v1'],

  'MICHAL7FT_V0': ['reach-v1', 'push-v1', 'button-press-v1', 'soccer-v1',
                   'shelf-place-v1', 'pick-place-v1', 'pick-place-wall-v1'],

  'PLATES': ['plate-slide-v1', 'plate-slide-side-v1', 'plate-slide-back-v1',
             'plate-slide-back-side-v1'],

  'HANDLES': ['handle-press-v1', 'handle-pull-v1', 'handle-press-side-v1',
              'handle-pull-side-v1'],

  'PRETTY_HARD_V0': ['coffee-pull-v1', 'shelf-place-v1', 'faucet-close-v1',
                     'handle-press-side-v1', 'push-wall-v1', 'sweep-v1',
                     'stick-push-v1', 'bin-picking-v1', 'basketball-v1',
                     'hammer-v1'],

  'HARDEST10_EASY_ORDER': ['stick-pull-v1', 'push-back-v1', 'lever-pull-v1',
                           'assembly-v1', 'disassemble-v1', 'pick-place-v1',
                           'peg-insert-side-v1', 'peg-unplug-side-v1',
                           'pick-place-wall-v1', 'pick-out-of-hole-v1'],

  'HARDEST10_HARD_ORDER': ['pick-place-v1', 'lever-pull-v1',
                           'peg-unplug-side-v1', 'peg-insert-side-v1',
                           'disassemble-v1', 'push-back-v1',
                           'pick-out-of-hole-v1', 'pick-place-wall-v1',
                           'stick-pull-v1', 'assembly-v1'],

  'MASHUP_V0': ['button-press-topdown-v1', 'push-v1', 'drawer-open-v1',
                'window-close-v1', 'pick-place-v1', 'peg-unplug-side-v1',
                'peg-insert-side-v1', 'push-back-v1', 'pick-place-wall-v1',
                'stick-pull-v1'],

  'MASHUP_RND_ORD_1': ['pick-place-wall-v1', 'window-close-v1', 'stick-pull-v1',
                       'peg-unplug-side-v1', 'button-press-topdown-v1',
                       'pick-place-v1', 'push-back-v1', 'peg-insert-side-v1',
                       'push-v1', 'drawer-open-v1'],

  'MASHUP_RND_ORD_2': ['stick-pull-v1', 'push-v1', 'button-press-topdown-v1',
                       'peg-unplug-side-v1', 'peg-insert-side-v1',
                       'drawer-open-v1', 'pick-place-v1', 'push-back-v1',
                       'window-close-v1', 'pick-place-wall-v1'],

  'PRETTY_MASHUP_ORD_1': ['hammer-v1', 'push-wall-v1', 'faucet-close-v1',
                          'push-back-v1', 'stick-pull-v1',
                          'handle-press-side-v1', 'push-v1', 'shelf-place-v1',
                          'window-close-v1', 'peg-unplug-side-v1'],

  'PRETTY_MASHUP_ORD_2': ['handle-press-side-v1', 'faucet-close-v1',
                          'shelf-place-v1', 'stick-pull-v1',
                          'peg-unplug-side-v1', 'hammer-v1', 'push-back-v1',
                          'push-wall-v1', 'push-v1', 'window-close-v1'],

  'PRETTY_MASHUP_ORD_3': ['stick-pull-v1', 'push-wall-v1', 'shelf-place-v1',
                          'window-close-v1', 'hammer-v1', 'peg-unplug-side-v1',
                          'push-back-v1', 'faucet-close-v1', 'push-v1',
                          'handle-press-side-v1']
}

task_seq_to_task_list['DOUBLE_MT10'] = (task_seq_to_task_list['MT10'] +
                                        task_seq_to_task_list['MT10'])

task_seq_to_task_list['DOUBLE_MASHUP_V0'] = (
  task_seq_to_task_list['MASHUP_V0'] + task_seq_to_task_list['MASHUP_V0'])

task_seq_to_task_list['DOUBLE_PMO1'] = (task_seq_to_task_list['PRETTY_MASHUP_ORD_1'] +
                                        task_seq_to_task_list['PRETTY_MASHUP_ORD_1'])
task_seq_to_task_list['DOUBLE_PMO2'] = (task_seq_to_task_list['PRETTY_MASHUP_ORD_2'] +
                                        task_seq_to_task_list['PRETTY_MASHUP_ORD_2'])
task_seq_to_task_list['DOUBLE_PMO3'] = (task_seq_to_task_list['PRETTY_MASHUP_ORD_3'] +
                                        task_seq_to_task_list['PRETTY_MASHUP_ORD_3'])

tasks_avg_return = {
  'reach-v1': 300297.0,
  'push-v1': 220606.0,
  'pick-place-v1': 185123.0,
  'door-open-v1': 152689.0,
  'drawer-open-v1': 131866.0,
  'drawer-close-v1': 211204.0,
  'button-press-topdown-v1': 247831.0,
  'peg-insert-side-v1': 149642.0,
  'window-open-v1': 111776.0,
  'window-close-v1': 89440.0,
}

task_reward_scales = {
  key: tasks_avg_return['reach-v1'] / item
  for key, item in tasks_avg_return.items()
}
